"""Multiprocessing execution for FELP frame jobs."""

from __future__ import annotations

from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    TimeoutError as FuturesTimeoutError,
    wait,
)
from concurrent.futures.process import BrokenProcessPool
from dataclasses import replace
import gc
import json
import multiprocessing as mp
import os
import sys
import threading
import time
import traceback
from typing import Callable, Iterable, Iterator, TypeVar

import numpy as np

from felp.conversion import StructureConversionService
from felp.core import (
    CVSpec,
    DihedralRestraintSpec,
    FrameResult,
    RunConfig,
    SnapshotJob,
    default_backbone_cv_specs,
    normalize_mutation_target_name,
)
from felp.energy import build_energy_evaluator
from felp.energy.evaluators import OpenMMEnergyEvaluator
from felp.perturbation import PyRosettaPerturbationProtocol
from felp.sidechain_ensemble import (
    attach_ensemble_to_frame,
    members_from_energy_results,
)
from felp.perturbation.validation import (
    ensure_measurement_matches_restraints,
    ensure_perturbation_matches_structure,
    ensure_pose_residue_matches_expected,
    measurement_deviation,
    target_cv_measurement_from_restraints,
)
from felp.protocol_summary import build_protocol_summary
from felp.restraints import BackboneDihedralCVProtocol
from felp.reweighting import (
    AssignmentPolicy,
    CVAssignmentPolicy,
    MicrostateAtlas,
    PhiPsiAssignmentPolicy,
)

_WORKER_CONFIG: RunConfig | None = None
_WORKER_ATLAS: MicrostateAtlas | None = None
_WORKER_CONVERSION_SERVICE: StructureConversionService | None = None
_WORKER_PERTURBATION_PROTOCOL: PyRosettaPerturbationProtocol | None = None
_WORKER_ENERGY_EVALUATOR = None
_WORKER_RESTRAINT_PROTOCOL: BackboneDihedralCVProtocol | None = None
_WORKER_TEMPLATE_STRUCTURE = None
_WORKER_ASSIGNMENT_POLICY: AssignmentPolicy | None = None
_WORKER_RUNTIME_PROTONATION_PRECHECKED: bool = False

_T = TypeVar("_T")


def _default_assignment_policy(config: RunConfig) -> AssignmentPolicy:
    """Choose the legacy or generalized assignment policy for this run."""

    if (
        not config.restraints.distance_cv_specs
        and config.restraints.cv_specs
        == default_backbone_cv_specs(config.perturbation.target_residue_number)
    ):
        return PhiPsiAssignmentPolicy()
    return CVAssignmentPolicy()


def _apply_cpu_thread_limits(config: RunConfig) -> None:
    """Constrain OpenMM CPU and OpenMP threading per worker.

    On many-core machines the OpenMM CPU platform will, by default, fan out a
    single tiny minimization across every available logical core. That has
    been seen to produce numerical instability (NaN coordinates) when stacked
    on top of FELP's per-frame multiprocessing. We always cap OpenMM CPU
    threads to ``cpu_threads`` (default 1) when the CPU platform is selected,
    and conservatively cap nested OpenMP/MKL parallelism in the same way
    unless the user has already pinned them explicitly.
    """

    if config.energy.platform_name != "CPU":
        return
    cpu_threads = max(1, int(config.energy.cpu_threads))
    os.environ["OPENMM_CPU_THREADS"] = str(cpu_threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(cpu_threads))


def initialize_worker(
    config: RunConfig,
    atlas: MicrostateAtlas,
    assignment_policy: AssignmentPolicy | None = None,
) -> None:
    """Initialize heavy dependencies (PyRosetta, OpenMM) exactly once per worker process."""
    init_start = time.perf_counter()
    global _WORKER_ATLAS
    global _WORKER_ASSIGNMENT_POLICY
    global _WORKER_CONFIG
    global _WORKER_CONVERSION_SERVICE
    global _WORKER_ENERGY_EVALUATOR
    global _WORKER_PERTURBATION_PROTOCOL
    global _WORKER_RESTRAINT_PROTOCOL
    global _WORKER_RUNTIME_PROTONATION_PRECHECKED
    global _WORKER_TEMPLATE_STRUCTURE

    _apply_cpu_thread_limits(config)
    _WORKER_CONFIG = config
    _WORKER_ATLAS = atlas
    _WORKER_ASSIGNMENT_POLICY = assignment_policy or _default_assignment_policy(config)
    _WORKER_RUNTIME_PROTONATION_PRECHECKED = False
    _WORKER_CONVERSION_SERVICE = StructureConversionService(
        pyrosetta_init_options=_build_pyrosetta_options(config),
        radii_set=config.input.radii_set,
    )
    _WORKER_TEMPLATE_STRUCTURE = _WORKER_CONVERSION_SERVICE.load_topology(
        config.input.topology_file
    )
    ensure_perturbation_matches_structure(_WORKER_TEMPLATE_STRUCTURE, config.perturbation)
    _WORKER_CONVERSION_SERVICE.validate_rosetta_structure_residue_types(
        _WORKER_TEMPLATE_STRUCTURE,
        context="the reference topology",
    )
    for mutation in config.perturbation.mutations:
        _WORKER_CONVERSION_SERVICE.validate_rosetta_residue_type(
            mutation.target_residue_name,
            context="mutation target",
        )
        if mutation.original_residue_name is not None:
            _WORKER_CONVERSION_SERVICE.validate_rosetta_residue_type(
                mutation.original_residue_name,
                context="reference/original state",
            )
    _WORKER_RESTRAINT_PROTOCOL = BackboneDihedralCVProtocol(
        cv_specs=config.restraints.cv_specs,
        distance_cv_specs=config.restraints.distance_cv_specs,
        dihedral_cv_specs=config.restraints.dihedral_cv_specs,
        difference_distance_cv_specs=config.restraints.difference_distance_cv_specs,
        force_constant=config.restraints.effective_rosetta_force_constant(),
    )
    perturbation_protocol_kwargs = {
        "conversion_service": _WORKER_CONVERSION_SERVICE,
        "mutation_residue_numbers": config.perturbation.mutation_residue_numbers,
        "score_function_name": config.rosetta.score_function,
        "perturbation_mode": config.rosetta.perturbation_mode,
        "relax_iterations": config.rosetta.relax_iterations,
        "repack_trials": config.rosetta.repack_trials,
        "cartesian": config.rosetta.cartesian,
        "mobility": config.mobility,
        "output_dir": config.output.output_dir,
        "write_pose_pdbs": config.output.write_pose_pdbs,
        "rosetta_distance_constraint_sd_angstrom": (
            config.restraints.rosetta_distance_constraint_sd_angstrom
        ),
        "distance_constraint_tolerance_angstrom": (
            config.restraints.distance_measurement_tolerance_angstrom
        ),
    }
    if config.restraints.rosetta_target_backbone_restraints:
        perturbation_protocol_kwargs["backbone_constraint_tolerance_degrees"] = (
            config.restraints.rosetta_target_backbone_restraint_tolerance_degrees
            or config.restraints.cv_measurement_tolerance_degrees
        )
    _WORKER_PERTURBATION_PROTOCOL = PyRosettaPerturbationProtocol(
        **perturbation_protocol_kwargs
    )
    try:
        _WORKER_ENERGY_EVALUATOR = build_energy_evaluator(
            config.energy,
            _WORKER_CONVERSION_SERVICE,
            mobility_config=config.mobility,
            mutation_residue_numbers=config.perturbation.mutation_residue_numbers,
        )
    except TypeError:
        _WORKER_ENERGY_EVALUATOR = build_energy_evaluator(
            config.energy,
            _WORKER_CONVERSION_SERVICE,
        )
    if config.parallel.progress_mode != "off":
        elapsed = time.perf_counter() - init_start
        print(
            f"[FELP worker {os.getpid()}] initialized in {elapsed:.1f}s",
            file=sys.stderr,
            flush=True,
        )


def _build_pyrosetta_options(config: RunConfig) -> str:
    options = config.rosetta.init_options.strip()
    if config.rosetta.database_path:
        options = f"{options} -database {config.rosetta.database_path}".strip()
    if (
        config.rosetta.perturbation_mode == "repack"
        and config.rosetta.repack_trials > 1
        and "-packing:ndruns" not in options
        and "-ndruns" not in options
    ):
        options = f"{options} -packing:ndruns {config.rosetta.repack_trials}".strip()
    protonation_names: set[str] = set()
    for mutation in config.perturbation.mutations:
        protonation_names.add(normalize_mutation_target_name(mutation.target_residue_name))
        if mutation.original_residue_name is not None:
            protonation_names.add(
                normalize_mutation_target_name(mutation.original_residue_name),
            )
    if "HIS" in protonation_names and "-no_optH" not in options and "-optH" not in options:
        options = f"{options} -no_optH false".strip()
    if (
        protonation_names & {"HIP", "ASH", "GLH", "CYM", "LYN"}
        and "-keep_input_protonation_state" not in options
    ):
        options = f"{options} -keep_input_protonation_state".strip()
    if protonation_names & {"HIP", "ASH", "GLH", "CYM", "LYN"} and "-pH_mode" not in options:
        options = f"{options} -pH_mode".strip()
    if config.rosetta.symmetric_gly_tables and "-symmetric_gly_tables" not in options:
        options = f"{options} -symmetric_gly_tables true".strip()
    if config.rosetta.mute_tracers and "-mute" not in options:
        options = f"{options} -mute all".strip()
    return options


def _measure_structure(structure):
    return _WORKER_RESTRAINT_PROTOCOL.measure_structure(structure)


def _debug_outputs_enabled(job: SnapshotJob) -> bool:
    if _WORKER_CONFIG is None or not _WORKER_CONFIG.output.debug_frame_outputs:
        return False
    selected = set(_WORKER_CONFIG.output.debug_frame_indices)
    return not selected or job.frame_index in selected


def _debug_frame_dir(job: SnapshotJob) -> str:
    if _WORKER_CONFIG is None:
        raise RuntimeError("Worker has not been initialized.")
    path = os.path.join(
        _WORKER_CONFIG.output.output_dir,
        "frame_debug",
        f"frame_{job.frame_index:05d}",
    )
    os.makedirs(path, exist_ok=True)
    return path


def _write_debug_structure(structure, directory: str, filename: str) -> str:
    path = os.path.join(directory, filename)
    structure.save(path, overwrite=True)
    return os.path.relpath(path, _WORKER_CONFIG.output.output_dir)


def _write_debug_json(directory: str, filename: str, payload: dict) -> str:
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return os.path.relpath(path, _WORKER_CONFIG.output.output_dir)


def _dihedral_restraint_key(spec: DihedralRestraintSpec) -> tuple:
    return (spec.kind, spec.residue_numbers, spec.atom_names)


def _rosetta_backbone_restraint_residue_numbers() -> tuple[int, ...]:
    if _WORKER_CONFIG is None:
        raise RuntimeError("Worker has not been initialized.")
    residue_numbers = _WORKER_CONFIG.restraints.rosetta_backbone_restraint_residue_numbers
    if residue_numbers:
        return residue_numbers
    return _WORKER_CONFIG.perturbation.mutation_residue_numbers


def _backbone_cv_exists_for_pose(pose, kind: str, residue_number: int) -> bool:
    total_residue = int(pose.total_residue())
    if kind == "phi":
        return 1 < residue_number <= total_residue
    if kind in {"psi", "omega"}:
        return 1 <= residue_number < total_residue
    return False


def _rosetta_target_backbone_restraints(reference_pose) -> tuple[DihedralRestraintSpec, ...]:
    if _WORKER_CONFIG is None:
        raise RuntimeError("Worker has not been initialized.")
    kinds = _WORKER_CONFIG.restraints.rosetta_target_backbone_restraints
    if not kinds:
        return ()
    cv_specs = tuple(
        CVSpec(
            kind=kind,
            residue_number=residue_number,
            name=f"rosetta_residue_{residue_number}_{kind}",
        )
        for residue_number in _rosetta_backbone_restraint_residue_numbers()
        for kind in kinds
        if _backbone_cv_exists_for_pose(reference_pose, kind, residue_number)
    )
    if not cv_specs:
        return ()
    protocol = BackboneDihedralCVProtocol(
        cv_specs=cv_specs,
        distance_cv_specs=(),
        force_constant=(
            _WORKER_CONFIG.restraints.effective_rosetta_target_backbone_force_constant()
        ),
    )
    return tuple(protocol.build_dihedral_restraints_from_reference_pose(reference_pose))


def _rosetta_preparation_restraints(reference_pose, restraint_specs) -> tuple:
    base_specs = tuple(restraint_specs)
    extra_specs = _rosetta_target_backbone_restraints(reference_pose)
    if not extra_specs:
        return base_specs
    existing_dihedral_keys = {
        _dihedral_restraint_key(spec)
        for spec in base_specs
        if isinstance(spec, DihedralRestraintSpec)
    }
    unique_extra_specs = tuple(
        spec
        for spec in extra_specs
        if _dihedral_restraint_key(spec) not in existing_dihedral_keys
    )
    return (*base_specs, *unique_extra_specs)


def _run_runtime_protonation_preflight(reference_pose) -> None:
    """Run one-time protonation/type checks on the first coordinate-bearing frame."""

    global _WORKER_RUNTIME_PROTONATION_PRECHECKED
    if _WORKER_RUNTIME_PROTONATION_PRECHECKED:
        return
    if _WORKER_CONFIG is None or _WORKER_CONVERSION_SERVICE is None:
        raise RuntimeError("Worker has not been initialized.")
    if _WORKER_PERTURBATION_PROTOCOL is None or _WORKER_ENERGY_EVALUATOR is None:
        raise RuntimeError("Worker has not been initialized.")

    mutations = tuple(_WORKER_CONFIG.perturbation.mutations)

    if any(spec.original_residue_name is not None for spec in mutations):
        for spec in mutations:
            if spec.original_residue_name is not None:
                ensure_pose_residue_matches_expected(
                    reference_pose,
                    spec.residue_number,
                    spec.original_residue_name,
                    context="reference-state setup",
                )
    # Use the real mutation path here so canonical mutations (e.g. GLY -> TYR)
    # are preflighted through the same code path as production frames, while
    # protonation/tautomer variants still resolve via mutate_pose().
    probe_perturbed_pose = _WORKER_PERTURBATION_PROTOCOL.mutate_pose(
        reference_pose,
        mutations,
        (),
    )
    for spec in mutations:
        ensure_pose_residue_matches_expected(
            probe_perturbed_pose,
            spec.residue_number,
            spec.target_residue_name,
            context="perturbed-state setup",
        )
    if _WORKER_CONFIG.energy.backend == "openmm":
        probe_reference_structure = _WORKER_CONVERSION_SERVICE.prepare_structure_for_openmm(
            _WORKER_CONVERSION_SERVICE.pose_to_structure(reference_pose)
        )
        probe_perturbed_structure = _WORKER_CONVERSION_SERVICE.prepare_structure_for_openmm(
            _WORKER_CONVERSION_SERVICE.pose_to_structure(probe_perturbed_pose)
        )
        _WORKER_ENERGY_EVALUATOR.validate_topology(probe_reference_structure)
        _WORKER_ENERGY_EVALUATOR.validate_topology(probe_perturbed_structure)
    _WORKER_RUNTIME_PROTONATION_PRECHECKED = True


def process_frame_job(job: SnapshotJob) -> FrameResult:
    """Process a single frame: assign microstate, convert, relax, perturb, rescore."""
    if any(
        item is None
        for item in (
            _WORKER_CONFIG,
            _WORKER_ATLAS,
            _WORKER_CONVERSION_SERVICE,
            _WORKER_PERTURBATION_PROTOCOL,
            _WORKER_RESTRAINT_PROTOCOL,
            _WORKER_TEMPLATE_STRUCTURE,
            _WORKER_ASSIGNMENT_POLICY,
        )
    ):
        raise RuntimeError("Worker has not been initialized.")

    coordinates = np.asarray(job.coordinates_angstrom, dtype=float)
    if coordinates.size == 0 or not np.isfinite(coordinates).all():
        raise ValueError(
            f"Frame {job.frame_index} has empty or non-finite coordinates "
            f"(trajectory_id={job.trajectory_id}, "
            f"source_frame_index={job.source_frame_index})."
        )

    structure = _WORKER_CONVERSION_SERVICE.clone_structure_with_coordinates(
        _WORKER_TEMPLATE_STRUCTURE,
        job.coordinates_angstrom,
    )
    reference_pose = _WORKER_CONVERSION_SERVICE.structure_to_pose(structure)
    reference_pose = _WORKER_PERTURBATION_PROTOCOL.enforce_residue_identity(
        reference_pose,
        _WORKER_CONFIG.perturbation.mutations,
    )
    _run_runtime_protonation_preflight(reference_pose)
    reference_measurement = _WORKER_RESTRAINT_PROTOCOL.measure_pose(reference_pose)
    assignment = _WORKER_ASSIGNMENT_POLICY.assign(
        _WORKER_ATLAS,
        job,
        reference_measurement,
    )
    restraint_specs = _WORKER_RESTRAINT_PROTOCOL.build_from_reference_pose(reference_pose)
    mutations = tuple(_WORKER_CONFIG.perturbation.mutations)
    rosetta_restraint_specs = _rosetta_preparation_restraints(
        reference_pose,
        restraint_specs,
    )
    cv_tol = _WORKER_CONFIG.restraints.cv_measurement_tolerance_degrees
    distance_tol = _WORKER_CONFIG.restraints.distance_measurement_tolerance_angstrom
    debug_outputs_enabled = _debug_outputs_enabled(job)
    debug_dir = _debug_frame_dir(job) if debug_outputs_enabled else None
    debug_artifacts: dict[str, str] = {}
    reference_structure = None
    perturbed_structure = None
    openmm_restraint_specs = ()

    if _WORKER_CONFIG.energy.backend == "openmm":
        openmm_fc = _WORKER_CONFIG.restraints.effective_openmm_force_constant()
        openmm_restraint_specs = tuple(
            replace(spec, force_constant=openmm_fc) for spec in restraint_specs
        )
        prepared_reference_pose = _WORKER_PERTURBATION_PROTOCOL.prepare_reference_pose(
            reference_pose,
            rosetta_restraint_specs,
            job.frame_index,
        )
        for spec in mutations:
            if spec.original_residue_name is not None:
                ensure_pose_residue_matches_expected(
                    prepared_reference_pose,
                    spec.residue_number,
                    spec.original_residue_name,
                    context="Rosetta-prepared reference state",
                )
        reference_structure = _WORKER_CONVERSION_SERVICE.pose_to_structure(
            prepared_reference_pose
        )
        reference_structure = _WORKER_CONVERSION_SERVICE.prepare_structure_for_openmm(
            reference_structure
        )
        perturbed_pose = _WORKER_PERTURBATION_PROTOCOL.apply_perturbation(
            prepared_reference_pose,
            mutations,
            rosetta_restraint_specs,
            job.frame_index,
        )
        for spec in mutations:
            ensure_pose_residue_matches_expected(
                perturbed_pose,
                spec.residue_number,
                spec.target_residue_name,
                context="Rosetta-relaxed perturbed state",
            )
        perturbed_structure = _WORKER_CONVERSION_SERVICE.pose_to_structure(
            perturbed_pose
        )
        perturbed_structure = _WORKER_CONVERSION_SERVICE.prepare_structure_for_openmm(
            perturbed_structure
        )
        if debug_dir is not None:
            debug_artifacts["reference_openmm_input_pdb"] = _write_debug_structure(
                reference_structure,
                debug_dir,
                "reference_openmm_input.pdb",
            )
            debug_artifacts["perturbed_openmm_input_pdb"] = _write_debug_structure(
                perturbed_structure,
                debug_dir,
                "perturbed_openmm_input.pdb",
            )
        reference_energy = _WORKER_ENERGY_EVALUATOR.evaluate(
            structure=reference_structure,
            restraint_specs=openmm_restraint_specs,
            collect_energy_components=debug_outputs_enabled,
        )
        protocol_summary = build_protocol_summary(
            _WORKER_CONFIG,
            restraint_specs=restraint_specs,
            rosetta_restraint_specs=rosetta_restraint_specs,
            openmm_restraint_specs=openmm_restraint_specs,
        )
        perturbed_energy = _WORKER_ENERGY_EVALUATOR.evaluate(
            structure=perturbed_structure,
            restraint_specs=openmm_restraint_specs,
            collect_energy_components=debug_outputs_enabled,
        )
        reference_minimized_structure = (
            _WORKER_CONVERSION_SERVICE.clone_structure_with_coordinates(
                reference_structure,
                reference_energy.minimized_coordinates_angstrom,
            )
        )
        perturbed_minimized_structure = (
            _WORKER_CONVERSION_SERVICE.clone_structure_with_coordinates(
                perturbed_structure,
                perturbed_energy.minimized_coordinates_angstrom,
            )
        )
        if debug_dir is not None:
            debug_artifacts["reference_openmm_minimized_pdb"] = _write_debug_structure(
                reference_minimized_structure,
                debug_dir,
                "reference_openmm_minimized.pdb",
            )
            debug_artifacts["perturbed_openmm_minimized_pdb"] = _write_debug_structure(
                perturbed_minimized_structure,
                debug_dir,
                "perturbed_openmm_minimized.pdb",
            )
        reference_relaxed_measurement = _measure_structure(
            reference_minimized_structure
        )
        perturbed_relaxed_measurement = _measure_structure(
            perturbed_minimized_structure
        )
        ensure_measurement_matches_restraints(
            reference_relaxed_measurement,
            restraint_specs,
            context="OpenMM-minimized reference state",
            tolerance_degrees=cv_tol,
            distance_tolerance_angstrom=distance_tol,
        )
        ensure_measurement_matches_restraints(
            perturbed_relaxed_measurement,
            restraint_specs,
            context="OpenMM-minimized perturbed state",
            tolerance_degrees=cv_tol,
            distance_tolerance_angstrom=distance_tol,
        )
    else:
        prepared_reference_pose = _WORKER_PERTURBATION_PROTOCOL.prepare_reference_pose(
            reference_pose,
            rosetta_restraint_specs,
            job.frame_index,
        )
        for spec in mutations:
            if spec.original_residue_name is not None:
                ensure_pose_residue_matches_expected(
                    prepared_reference_pose,
                    spec.residue_number,
                    spec.original_residue_name,
                    context="Rosetta-relaxed reference state",
                )
        perturbed_pose = _WORKER_PERTURBATION_PROTOCOL.apply_perturbation(
            prepared_reference_pose,
            mutations,
            rosetta_restraint_specs,
            job.frame_index,
        )
        for spec in mutations:
            ensure_pose_residue_matches_expected(
                perturbed_pose,
                spec.residue_number,
                spec.target_residue_name,
                context="Rosetta-relaxed perturbed state",
            )
        reference_relaxed_measurement = _WORKER_RESTRAINT_PROTOCOL.measure_pose(
            prepared_reference_pose
        )
        perturbed_relaxed_measurement = _WORKER_RESTRAINT_PROTOCOL.measure_pose(
            perturbed_pose
        )
        ensure_measurement_matches_restraints(
            reference_relaxed_measurement,
            restraint_specs,
            context="Rosetta-relaxed reference state",
            tolerance_degrees=cv_tol,
            distance_tolerance_angstrom=distance_tol,
        )
        ensure_measurement_matches_restraints(
            perturbed_relaxed_measurement,
            restraint_specs,
            context="Rosetta-relaxed perturbed state",
            tolerance_degrees=cv_tol,
            distance_tolerance_angstrom=distance_tol,
        )
        reference_energy = _WORKER_ENERGY_EVALUATOR.evaluate(
            pose=prepared_reference_pose,
            restraint_specs=restraint_specs,
            collect_energy_components=debug_outputs_enabled,
        )
        perturbed_energy = _WORKER_ENERGY_EVALUATOR.evaluate(
            pose=perturbed_pose,
            restraint_specs=restraint_specs,
            collect_energy_components=debug_outputs_enabled,
        )
        protocol_summary = build_protocol_summary(
            _WORKER_CONFIG,
            restraint_specs=restraint_specs,
            rosetta_restraint_specs=rosetta_restraint_specs,
            openmm_restraint_specs=(),
        )

    delta_u = perturbed_energy.energy_kj_mol - reference_energy.energy_kj_mol
    target_measurement = target_cv_measurement_from_restraints(restraint_specs)
    reference_deviation = measurement_deviation(
        reference_relaxed_measurement,
        restraint_specs,
    )
    perturbed_deviation = measurement_deviation(
        perturbed_relaxed_measurement,
        restraint_specs,
    )
    metadata = {
        "energy_backend": _WORKER_CONFIG.energy.backend,
        "assignment_measurements": assignment.measurements,
        "cv_centers": assignment.cv_centers,
        "phi_center": assignment.phi_center,
        "psi_center": assignment.psi_center,
        "plot_x": assignment.plot_x,
        "plot_y": assignment.plot_y,
        "target_cv_values_degrees": target_measurement.as_dict(),
        "reference_cv_values_degrees": reference_relaxed_measurement.as_dict(),
        "perturbed_cv_values_degrees": perturbed_relaxed_measurement.as_dict(),
        "reference_cv_deviation_degrees": reference_deviation,
        "perturbed_cv_deviation_degrees": perturbed_deviation,
        "target_cv_values": target_measurement.as_dict(),
        "reference_cv_values": reference_relaxed_measurement.as_dict(),
        "perturbed_cv_values": perturbed_relaxed_measurement.as_dict(),
        "reference_cv_deviation": reference_deviation,
        "perturbed_cv_deviation": perturbed_deviation,
        "mobility": _WORKER_CONFIG.mobility.__dict__,
        "protocol": protocol_summary,
    }
    if target_measurement.phi_degrees is not None:
        metadata["target_phi_degrees"] = target_measurement.phi_degrees
    if target_measurement.psi_degrees is not None:
        metadata["target_psi_degrees"] = target_measurement.psi_degrees
    if "phi" in reference_deviation:
        metadata["reference_phi_deviation_degrees"] = reference_deviation["phi"]
    if "psi" in reference_deviation:
        metadata["reference_psi_deviation_degrees"] = reference_deviation["psi"]
    if "phi" in perturbed_deviation:
        metadata["perturbed_phi_deviation_degrees"] = perturbed_deviation["phi"]
    if "psi" in perturbed_deviation:
        metadata["perturbed_psi_deviation_degrees"] = perturbed_deviation["psi"]
    if debug_dir is not None:
        debug_artifacts["energy_components_json"] = _write_debug_json(
            debug_dir,
            "energy_components.json",
            {
                "frame_index": job.frame_index,
                "source_frame_index": job.source_frame_index,
                "trajectory_id": job.trajectory_id,
                "microstate_id": assignment.state_id,
                "microstate_label": assignment.label,
                "energy_backend": _WORKER_CONFIG.energy.backend,
                "reference_energy_kj_mol": reference_energy.energy_kj_mol,
                "perturbed_energy_kj_mol": perturbed_energy.energy_kj_mol,
                "delta_u_kj_mol": delta_u,
                "reference_energy_metadata": reference_energy.metadata,
                "perturbed_energy_metadata": perturbed_energy.metadata,
                "protocol": protocol_summary,
                "target_cv_values": target_measurement.as_dict(),
                "reference_cv_values": reference_relaxed_measurement.as_dict(),
                "perturbed_cv_values": perturbed_relaxed_measurement.as_dict(),
                "reference_cv_deviation": reference_deviation,
                "perturbed_cv_deviation": perturbed_deviation,
            },
        )
        metadata["debug_artifacts"] = dict(debug_artifacts)

    frame = FrameResult(
        frame_index=job.frame_index,
        stride_number=job.stride_number,
        microstate_id=assignment.state_id,
        microstate_label=assignment.label,
        phi_degrees=assignment.phi_degrees,
        psi_degrees=assignment.psi_degrees,
        reference_energy_kj_mol=reference_energy.energy_kj_mol,
        perturbed_energy_kj_mol=perturbed_energy.energy_kj_mol,
        delta_u_kj_mol=delta_u,
        reference_relaxed_phi_degrees=reference_relaxed_measurement.phi_degrees,
        reference_relaxed_psi_degrees=reference_relaxed_measurement.psi_degrees,
        perturbed_relaxed_phi_degrees=perturbed_relaxed_measurement.phi_degrees,
        perturbed_relaxed_psi_degrees=perturbed_relaxed_measurement.psi_degrees,
        trajectory_id=job.trajectory_id,
        source_frame_index=job.source_frame_index,
        assignment_source=_WORKER_ASSIGNMENT_POLICY.name,
        state_level=assignment.state_level,
        metadata=metadata,
    )
    return _maybe_attach_sidechain_ensemble(
        frame,
        prepared_reference_pose=prepared_reference_pose,
        perturbed_pose=perturbed_pose,
        reference_structure=(
            reference_structure if _WORKER_CONFIG.energy.backend == "openmm" else None
        ),
        perturbed_structure=(
            perturbed_structure if _WORKER_CONFIG.energy.backend == "openmm" else None
        ),
        restraint_specs=(
            openmm_restraint_specs
            if _WORKER_CONFIG.energy.backend == "openmm"
            else restraint_specs
        ),
    )


def _structures_from_sampled_poses(sampled_poses):
    structures = []
    for sample in sampled_poses:
        structure = _WORKER_CONVERSION_SERVICE.pose_to_structure(sample.pose)
        structure = _WORKER_CONVERSION_SERVICE.prepare_structure_for_openmm(structure)
        structures.append(structure)
    return structures


def _maybe_attach_sidechain_ensemble(
    frame: FrameResult,
    *,
    prepared_reference_pose,
    perturbed_pose,
    reference_structure,
    perturbed_structure,
    restraint_specs,
) -> FrameResult:
    config = _WORKER_CONFIG.sidechain_ensemble
    if not config.enabled:
        return frame
    if _WORKER_CONFIG.energy.backend != "openmm":
        raise RuntimeError(
            "Sidechain-ensemble modes require the OpenMM energy backend."
        )
    if not isinstance(_WORKER_ENERGY_EVALUATOR, OpenMMEnergyEvaluator):
        raise RuntimeError("Sidechain-ensemble evaluation requires OpenMMEnergyEvaluator.")
    protocol = _WORKER_PERTURBATION_PROTOCOL
    sampled_reference = protocol.sample_sidechain_ensemble(
        prepared_reference_pose,
        config=config,
        branch="reference",
        restraint_specs=restraint_specs,
    )
    sampled_perturbed = protocol.sample_sidechain_ensemble(
        perturbed_pose,
        config=config,
        branch="perturbed",
        restraint_specs=restraint_specs,
    )
    reference_structures = _structures_from_sampled_poses(sampled_reference)
    perturbed_structures = _structures_from_sampled_poses(sampled_perturbed)
    reference_energies = _WORKER_ENERGY_EVALUATOR.evaluate_batch(
        reference_structures,
        restraint_specs,
        template_structure=reference_structure,
    )
    perturbed_energies = _WORKER_ENERGY_EVALUATOR.evaluate_batch(
        perturbed_structures,
        restraint_specs,
        template_structure=perturbed_structure,
    )
    members = members_from_energy_results(
        [*sampled_reference, *sampled_perturbed],
        [*reference_energies, *perturbed_energies],
        frame_index=frame.frame_index,
    )
    return attach_ensemble_to_frame(
        frame,
        members,
        config=config,
        beta=_WORKER_CONFIG.beta(),
    )


def _failed_frame_result(job: SnapshotJob, exc: BaseException) -> FrameResult:
    """Return a structured failure row so a single bad frame cannot abort a whole batch."""

    tb = traceback.format_exc()
    if len(tb) > 8000:
        tb = f"{tb[:8000]}\n... (traceback truncated)"

    nan = float("nan")
    return FrameResult(
        frame_index=job.frame_index,
        stride_number=job.stride_number,
        microstate_id="failed",
        microstate_label="failed",
        phi_degrees=None,
        psi_degrees=None,
        reference_energy_kj_mol=nan,
        perturbed_energy_kj_mol=nan,
        delta_u_kj_mol=nan,
        reference_relaxed_phi_degrees=None,
        reference_relaxed_psi_degrees=None,
        perturbed_relaxed_phi_degrees=None,
        perturbed_relaxed_psi_degrees=None,
        trajectory_id=job.trajectory_id,
        source_frame_index=job.source_frame_index,
        assignment_source="failed",
        state_level="failed",
        metadata={
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": tb,
        },
    )


def safe_process_frame_job(job: SnapshotJob) -> FrameResult:
    """Like `process_frame_job`, but never raises: failures become `FrameResult` rows."""

    try:
        return process_frame_job(job)
    except BaseException as exc:
        return _failed_frame_result(job, exc)


def _should_show_progress_bar(config: RunConfig, job_count: int) -> bool:
    if config.parallel.progress_mode == "on":
        return True
    if config.parallel.progress_mode == "off":
        return False
    # auto
    if job_count <= 1:
        return False
    if config.parallel.workers > 1:
        return True
    try:
        from tqdm.auto import tqdm as _tqdm  # type: ignore[import-not-found]

        _ = _tqdm  # noqa: F841
    except Exception:
        return False
    return job_count >= 50


def _maybe_tqdm(
    iterable: Iterable[_T], *, total: int, desc: str, enabled: bool
) -> Iterator[_T]:
    if not enabled:
        yield from iterable
        return

    try:
        from tqdm.auto import tqdm

        yield from tqdm(
            iterable,
            total=total,
            desc=desc,
            leave=True,
            dynamic_ncols=True,
            file=sys.stderr,
        )
        return
    except Exception:
        pass

    for index, item in enumerate(iterable, start=1):
        if index == 1 or index == total or index % max(total // 50, 1) == 0:
            print(f"{desc}: {index}/{total}", file=sys.stderr, flush=True)
        yield item


def _batched(items: list[_T], size: int) -> list[list[_T]]:
    """Split ``items`` into non-empty batches of at most ``size``."""

    batch_size = max(1, int(size))
    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]


def process_frame_job_batch(jobs: list[SnapshotJob]) -> list[FrameResult]:
    """Process a batch of frame jobs in one worker task."""

    return [safe_process_frame_job(job) for job in jobs]


def _default_in_flight_frame_count(worker_count: int) -> int:
    """Bound queued work while keeping every worker continuously supplied."""

    return max(64, min(2048, max(1, int(worker_count)) * 4))


class WorkerTerminated(RuntimeError):
    """Raised as a recorded frame failure when a worker process is killed."""


_POOL_START_EXCEPTIONS = (
    BrokenProcessPool,
    OSError,
    ValueError,
    RuntimeError,
)
_POOL_FAILURE_EXCEPTIONS = (
    BrokenProcessPool,
    FuturesTimeoutError,
    TimeoutError,
    OSError,
)
_MAX_INFRA_RETRIES = 1
_POOL_SHUTDOWN_TIMEOUT_SECONDS = 30


def reap_worker_processes(*, grace_seconds: float = 10.0) -> int:
    """Terminate leftover multiprocessing children so the parent can exit.

    After a long OpenMM/Rosetta pool run, interpreter shutdown can block
    forever joining spawn workers. That leaves a SLURM allocation occupied
    with no progress. Call this before returning from ``felp.py run``.
    """

    children = mp.active_children()
    for child in children:
        try:
            child.terminate()
        except Exception:
            continue
    deadline = time.monotonic() + max(0.0, float(grace_seconds))
    for child in children:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            child.join(remaining)
        except Exception:
            pass
        if child.is_alive():
            try:
                child.kill()
            except Exception:
                pass
            try:
                child.join(1.0)
            except Exception:
                pass
    return len(children)


def _shutdown_executor(executor: ProcessPoolExecutor | None, *, wait_for_workers: bool) -> None:
    """Shut down a pool, then kill anything that does not leave in time."""

    if executor is None:
        return
    # Capture the pool children before ``shutdown`` drops the executor's
    # references to them.  This is essential on the failure path: otherwise
    # spawn workers can become PPID-1 orphans and keep a SLURM allocation alive.
    children = mp.active_children()

    def _call() -> None:
        try:
            executor.shutdown(wait=wait_for_workers, cancel_futures=True)
        except TypeError:
            try:
                executor.shutdown(wait=wait_for_workers)
            except Exception:
                pass
        except Exception:
            pass

    if wait_for_workers:
        thread = threading.Thread(
            target=_call, name="felp-pool-shutdown", daemon=True
        )
        thread.start()
        thread.join(_POOL_SHUTDOWN_TIMEOUT_SECONDS)
        if thread.is_alive():
            print(
                f"[FELP] worker pool shutdown exceeded "
                f"{_POOL_SHUTDOWN_TIMEOUT_SECONDS:.0f}s; killing leftover workers.",
                file=sys.stderr,
                flush=True,
            )
    else:
        _call()

    for child in children:
        if not child.is_alive():
            continue
        try:
            child.terminate()
        except Exception:
            pass
    reap_worker_processes()


def _skip_unfinished_results(
    jobs: list[SnapshotJob],
    exc: BaseException,
) -> list[FrameResult]:
    """Turn leftover jobs into failed rows so the ensemble can finish."""

    return [
        _failed_frame_result(
            job,
            WorkerTerminated(
                "Skipped after a worker or pool failure while processing "
                f"frame {job.frame_index} (trajectory_id={job.trajectory_id}, "
                f"source_frame_index={job.source_frame_index}): "
                f"{type(exc).__name__}: {exc}"
            ),
        )
        for job in jobs
    ]


class FrameExecutor:
    """Dispatch frame jobs either serially or with multiprocessing."""

    def __init__(
        self,
        config: RunConfig,
        atlas: MicrostateAtlas,
        assignment_policy: AssignmentPolicy | None = None,
    ) -> None:
        self.config = config
        self.atlas = atlas
        self.assignment_policy = assignment_policy or _default_assignment_policy(
            config
        )

    def run(
        self,
        jobs: Iterable[SnapshotJob],
        on_checkpoint: Callable[[list[FrameResult]], None] | None = None,
    ) -> list[FrameResult]:
        """Execute jobs, surviving killed workers and checkpointing completed frames."""

        jobs = list(jobs)
        if not jobs:
            return []

        show_bar = _should_show_progress_bar(self.config, len(jobs))
        if self.config.parallel.workers == 1:
            return self._run_serial(jobs, show_bar=show_bar, on_checkpoint=on_checkpoint)

        pbar = None
        if show_bar:
            try:
                from tqdm.auto import tqdm

                pbar = tqdm(
                    total=len(jobs),
                    desc="frames",
                    leave=True,
                    dynamic_ncols=True,
                    file=sys.stderr,
                    unit="fr",
                )
            except Exception:
                pbar = None

        completed: list[FrameResult] = []
        remaining = list(jobs)
        infrastructure_attempts: dict[int, int] = {}
        consecutive_spawn_failures = 0
        executor: ProcessPoolExecutor | None = None

        def _record(new_results: list[FrameResult]) -> None:
            if not new_results:
                return
            completed.extend(new_results)
            self._maybe_fail_fast(new_results)
            if on_checkpoint is None:
                return
            try:
                on_checkpoint(list(new_results))
            except Exception as exc:
                print(
                    f"[FELP] checkpoint write failed ({type(exc).__name__}: {exc}); "
                    "continuing with in-memory results.",
                    file=sys.stderr,
                    flush=True,
                )

        def _skip(jobs_to_skip: list[SnapshotJob], exc: BaseException) -> None:
            if not jobs_to_skip:
                return
            indices = [job.frame_index for job in jobs_to_skip]
            preview = ", ".join(str(index) for index in indices[:12])
            if len(indices) > 12:
                preview = f"{preview}, ... ({len(indices)} frames)"
            print(
                f"[FELP] skipping {len(jobs_to_skip)} unfinished frame(s) after "
                f"{type(exc).__name__}: {exc}. Logged as failed and continuing. "
                f"indices={preview}",
                file=sys.stderr,
                flush=True,
            )
            skipped = _skip_unfinished_results(jobs_to_skip, exc)
            if pbar is not None:
                pbar.update(len(skipped))
            _record(skipped)

        def _drop_pool() -> None:
            nonlocal executor
            _shutdown_executor(executor, wait_for_workers=False)
            executor = None
            gc.collect()

        try:
            while remaining:
                if executor is None:
                    try:
                        executor = self._make_pool()
                    except _POOL_START_EXCEPTIONS as exc:
                        consecutive_spawn_failures += 1
                        if consecutive_spawn_failures >= 2:
                            _skip(
                                remaining,
                                exc
                                or WorkerTerminated(
                                    "Could not start another worker pool; remaining "
                                    "frames were logged as failed."
                                ),
                            )
                            remaining = []
                            break
                        continue
                try:
                    (
                        stream_results,
                        interrupted_jobs,
                        unsubmitted_jobs,
                        pool_error,
                    ) = (
                        self._submit_pool_stream(
                            remaining,
                            executor=executor,
                            pbar=pbar,
                            on_checkpoint=_record,
                        )
                    )
                except (KeyboardInterrupt, SystemExit):
                    raise
                except RuntimeError as exc:
                    if self.config.parallel.fail_fast:
                        raise
                    stream_results, interrupted_jobs, unsubmitted_jobs, pool_error = (
                        [],
                        list(remaining),
                        [],
                        exc,
                    )
                except BaseException as exc:
                    stream_results, interrupted_jobs, unsubmitted_jobs, pool_error = (
                        [],
                        list(remaining),
                        [],
                        exc,
                    )
                if pool_error is not None:
                    _drop_pool()

                retry_jobs: list[SnapshotJob] = []
                exhausted_jobs: list[SnapshotJob] = []
                for job in interrupted_jobs:
                    attempt = infrastructure_attempts.get(job.frame_index, 0) + 1
                    infrastructure_attempts[job.frame_index] = attempt
                    if attempt <= _MAX_INFRA_RETRIES:
                        retry_jobs.append(job)
                    else:
                        exhausted_jobs.append(job)
                if exhausted_jobs:
                    _skip(
                        exhausted_jobs,
                        pool_error
                        or WorkerTerminated("worker pool repeatedly interrupted"),
                    )
                remaining = retry_jobs + unsubmitted_jobs

                if stream_results or interrupted_jobs or pool_error is None:
                    consecutive_spawn_failures = 0
                    continue
                consecutive_spawn_failures += 1
                if consecutive_spawn_failures >= 2:
                    _skip(
                        remaining,
                        pool_error
                        or WorkerTerminated(
                            "Could not start another worker pool; remaining frames "
                            "were logged as failed."
                        ),
                    )
                    remaining = []
                    break
        finally:
            _shutdown_executor(executor, wait_for_workers=True)
            if pbar is not None:
                pbar.close()

        return sorted(completed, key=lambda item: item.frame_index)

    def _maybe_fail_fast(self, results: list[FrameResult]) -> None:
        if not self.config.parallel.fail_fast:
            return
        failed_result = next(
            (
                result
                for result in results
                if result.metadata.get("status") == "failed"
            ),
            None,
        )
        if failed_result is None:
            return
        raise RuntimeError(
            f"Frame {failed_result.frame_index} failed while --fail-fast was enabled: "
            f"{failed_result.metadata.get('error_type')}: "
            f"{failed_result.metadata.get('error_message')}"
        )

    def _run_serial(
        self,
        jobs: list[SnapshotJob],
        *,
        show_bar: bool,
        on_checkpoint: Callable[[list[FrameResult]], None] | None,
    ) -> list[FrameResult]:
        initialize_worker(self.config, self.atlas, self.assignment_policy)
        results: list[FrameResult] = []
        pending_checkpoint: list[FrameResult] = []
        checkpoint_every = 64
        for index, job in enumerate(
            _maybe_tqdm(jobs, total=len(jobs), desc="frames", enabled=show_bar),
            start=1,
        ):
            result = safe_process_frame_job(job)
            results.append(result)
            pending_checkpoint.append(result)
            if on_checkpoint is not None and (
                index == len(jobs) or len(pending_checkpoint) >= checkpoint_every
            ):
                on_checkpoint(list(pending_checkpoint))
                pending_checkpoint = []
            self._maybe_fail_fast([result])
        return results

    def _make_pool(self) -> ProcessPoolExecutor:
        context = mp.get_context(self.config.parallel.start_method)
        return ProcessPoolExecutor(
            max_workers=max(1, int(self.config.parallel.workers)),
            mp_context=context,
            initializer=initialize_worker,
            initargs=(self.config, self.atlas, self.assignment_policy),
        )

    def _submit_pool_stream(
        self,
        jobs: list[SnapshotJob],
        *,
        executor: ProcessPoolExecutor,
        pbar,
        on_checkpoint: Callable[[list[FrameResult]], None] | None = None,
    ) -> tuple[
        list[FrameResult],
        list[SnapshotJob],
        list[SnapshotJob],
        BaseException | None,
    ]:
        """Continuously refill a bounded pool queue until completion or pool failure.

        Submitted jobs interrupted by a dead or stalled pool are returned
        separately from jobs that were never submitted.  The caller retries
        interrupted jobs once before recording permanent failures.
        """

        if not jobs:
            return [], [], [], None

        job_batches = _batched(jobs, self.config.parallel.chunk_size)
        chunk_size = max(1, int(self.config.parallel.chunk_size))
        max_in_flight = max(
            1,
            _default_in_flight_frame_count(self.config.parallel.workers)
            // chunk_size,
        )
        next_batch_index = 0
        future_to_batch: dict = {}
        results: list[FrameResult] = []
        pending_checkpoint: list[FrameResult] = []
        checkpoint_every = 2048

        def _flush_checkpoint(force: bool = False) -> None:
            if on_checkpoint is None or not pending_checkpoint:
                return
            if not force and len(pending_checkpoint) < checkpoint_every:
                return
            on_checkpoint(list(pending_checkpoint))
            pending_checkpoint.clear()

        def _fail_batches(
            batches: Iterable[list[SnapshotJob]],
            exc: BaseException,
        ) -> None:
            for batch in batches:
                failed = _skip_unfinished_results(batch, exc)
                results.extend(failed)
                pending_checkpoint.extend(failed)
                if pbar is not None:
                    pbar.update(len(failed))

        def _unsubmitted_jobs() -> list[SnapshotJob]:
            return [
                job
                for batch in job_batches[next_batch_index:]
                for job in batch
            ]

        def _interrupted_jobs(
            batches: Iterable[list[SnapshotJob]],
        ) -> list[SnapshotJob]:
            return [job for batch in batches for job in batch]

        def _submit_available() -> None:
            nonlocal next_batch_index
            while (
                len(future_to_batch) < max_in_flight
                and next_batch_index < len(job_batches)
            ):
                batch = job_batches[next_batch_index]
                future = executor.submit(process_frame_job_batch, batch)
                future_to_batch[future] = batch
                next_batch_index += 1

        try:
            _submit_available()
            while future_to_batch:
                done, _ = wait(
                    set(future_to_batch),
                    return_when=FIRST_COMPLETED,
                )

                pool_error: BaseException | None = None
                interrupted_batches: list[list[SnapshotJob]] = []
                for future in done:
                    batch = future_to_batch.pop(future)
                    try:
                        batch_results = future.result(timeout=0)
                    except (BrokenProcessPool, FuturesTimeoutError) as exc:
                        pool_error = pool_error or exc
                        interrupted_batches.append(batch)
                    except Exception as exc:
                        _fail_batches([batch], exc)
                    else:
                        results.extend(batch_results)
                        pending_checkpoint.extend(batch_results)
                        if pbar is not None:
                            pbar.update(len(batch_results))

                if pool_error is not None:
                    for future in future_to_batch:
                        future.cancel()
                    interrupted_batches.extend(future_to_batch.values())
                    _flush_checkpoint(force=True)
                    return (
                        results,
                        _interrupted_jobs(interrupted_batches),
                        _unsubmitted_jobs(),
                        pool_error,
                    )

                _flush_checkpoint()
                _submit_available()

            _flush_checkpoint(force=True)
            return results, [], [], None
        except RuntimeError as exc:
            if bool(getattr(self.config.parallel, "fail_fast", False)):
                raise
            for future in future_to_batch:
                future.cancel()
            _flush_checkpoint(force=True)
            return (
                results,
                _interrupted_jobs(future_to_batch.values()),
                _unsubmitted_jobs(),
                exc,
            )
        except _POOL_FAILURE_EXCEPTIONS as exc:
            for future in future_to_batch:
                future.cancel()
            _flush_checkpoint(force=True)
            return (
                results,
                _interrupted_jobs(future_to_batch.values()),
                _unsubmitted_jobs(),
                exc,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            for future in future_to_batch:
                future.cancel()
            _flush_checkpoint(force=True)
            return (
                results,
                _interrupted_jobs(future_to_batch.values()),
                _unsubmitted_jobs(),
                exc,
            )
