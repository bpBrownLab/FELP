"""Multiprocessing execution for FELP frame jobs."""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import replace
import multiprocessing as mp
import os
import sys
import traceback
from typing import Iterable, Iterator, TypeVar

from felp.conversion import StructureConversionService
from felp.core import (
    FrameResult,
    RunConfig,
    SnapshotJob,
    default_backbone_cv_specs,
    normalize_mutation_target_name,
)
from felp.energy import build_energy_evaluator
from felp.perturbation import PyRosettaPerturbationProtocol
from felp.perturbation.validation import (
    ensure_measurement_matches_restraints,
    ensure_perturbation_matches_structure,
    ensure_pose_residue_matches_expected,
    measurement_deviation,
    target_cv_measurement_from_restraints,
)
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
    _WORKER_CONVERSION_SERVICE.validate_rosetta_residue_type(
        config.perturbation.target_residue_name,
        context="mutation target",
    )
    if config.perturbation.original_residue_name is not None:
        _WORKER_CONVERSION_SERVICE.validate_rosetta_residue_type(
            config.perturbation.original_residue_name,
            context="reference/original state",
        )
    _WORKER_RESTRAINT_PROTOCOL = BackboneDihedralCVProtocol(
        cv_specs=config.restraints.cv_specs,
        distance_cv_specs=config.restraints.distance_cv_specs,
        force_constant=config.restraints.effective_rosetta_force_constant(),
    )
    _WORKER_PERTURBATION_PROTOCOL = PyRosettaPerturbationProtocol(
        conversion_service=_WORKER_CONVERSION_SERVICE,
        target_residue_number=config.perturbation.target_residue_number,
        score_function_name=config.rosetta.score_function,
        perturbation_mode=config.rosetta.perturbation_mode,
        relax_iterations=config.rosetta.relax_iterations,
        repack_trials=config.rosetta.repack_trials,
        cartesian=config.rosetta.cartesian,
        mobility=config.mobility,
        output_dir=config.output.output_dir,
        write_pose_pdbs=config.output.write_pose_pdbs,
        rosetta_distance_constraint_sd_angstrom=(
            config.restraints.rosetta_distance_constraint_sd_angstrom
        ),
    )
    try:
        _WORKER_ENERGY_EVALUATOR = build_energy_evaluator(
            config.energy,
            _WORKER_CONVERSION_SERVICE,
            mobility_config=config.mobility,
            target_residue_number=config.perturbation.target_residue_number,
        )
    except TypeError:
        _WORKER_ENERGY_EVALUATOR = build_energy_evaluator(
            config.energy,
            _WORKER_CONVERSION_SERVICE,
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
    protonation_names = {
        normalize_mutation_target_name(name)
        for name in (
            config.perturbation.original_residue_name,
            config.perturbation.target_residue_name,
        )
        if name is not None
    }
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


def _run_runtime_protonation_preflight(reference_pose) -> None:
    """Run one-time protonation/type checks on the first coordinate-bearing frame."""

    global _WORKER_RUNTIME_PROTONATION_PRECHECKED
    if _WORKER_RUNTIME_PROTONATION_PRECHECKED:
        return
    if _WORKER_CONFIG is None or _WORKER_CONVERSION_SERVICE is None:
        raise RuntimeError("Worker has not been initialized.")
    if _WORKER_PERTURBATION_PROTOCOL is None or _WORKER_ENERGY_EVALUATOR is None:
        raise RuntimeError("Worker has not been initialized.")

    if _WORKER_CONFIG.perturbation.original_residue_name is not None:
        ensure_pose_residue_matches_expected(
            reference_pose,
            _WORKER_CONFIG.perturbation.target_residue_number,
            _WORKER_CONFIG.perturbation.original_residue_name,
            context="reference-state setup",
        )
    # Use the real mutation path here so canonical mutations (e.g. GLY -> TYR)
    # are preflighted through the same code path as production frames, while
    # protonation/tautomer variants still resolve via mutate_pose().
    probe_perturbed_pose = _WORKER_PERTURBATION_PROTOCOL.mutate_pose(
        reference_pose,
        _WORKER_CONFIG.perturbation.target_residue_number,
        _WORKER_CONFIG.perturbation.target_residue_name,
        (),
    )
    ensure_pose_residue_matches_expected(
        probe_perturbed_pose,
        _WORKER_CONFIG.perturbation.target_residue_number,
        _WORKER_CONFIG.perturbation.target_residue_name,
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

    structure = _WORKER_CONVERSION_SERVICE.clone_structure_with_coordinates(
        _WORKER_TEMPLATE_STRUCTURE,
        job.coordinates_angstrom,
    )
    reference_pose = _WORKER_CONVERSION_SERVICE.structure_to_pose(structure)
    reference_pose = _WORKER_PERTURBATION_PROTOCOL.enforce_residue_identity(
        reference_pose,
        _WORKER_CONFIG.perturbation.target_residue_number,
        _WORKER_CONFIG.perturbation.original_residue_name,
    )
    _run_runtime_protonation_preflight(reference_pose)
    reference_measurement = _WORKER_RESTRAINT_PROTOCOL.measure_pose(reference_pose)
    assignment = _WORKER_ASSIGNMENT_POLICY.assign(
        _WORKER_ATLAS,
        job,
        reference_measurement,
    )
    restraint_specs = _WORKER_RESTRAINT_PROTOCOL.build_from_reference_pose(reference_pose)
    cv_tol = _WORKER_CONFIG.restraints.cv_measurement_tolerance_degrees
    distance_tol = _WORKER_CONFIG.restraints.distance_measurement_tolerance_angstrom

    if _WORKER_CONFIG.energy.backend == "openmm":
        openmm_fc = _WORKER_CONFIG.restraints.effective_openmm_force_constant()
        openmm_restraint_specs = tuple(
            replace(spec, force_constant=openmm_fc) for spec in restraint_specs
        )
        reference_structure = _WORKER_CONVERSION_SERVICE.pose_to_structure(
            reference_pose
        )
        reference_structure = _WORKER_CONVERSION_SERVICE.prepare_structure_for_openmm(
            reference_structure
        )
        perturbed_pose = _WORKER_PERTURBATION_PROTOCOL.apply_perturbation(
            reference_pose,
            _WORKER_CONFIG.perturbation.target_residue_number,
            _WORKER_CONFIG.perturbation.target_residue_name,
            restraint_specs,
            job.frame_index,
        )
        ensure_pose_residue_matches_expected(
            perturbed_pose,
            _WORKER_CONFIG.perturbation.target_residue_number,
            _WORKER_CONFIG.perturbation.target_residue_name,
            context="Rosetta-relaxed perturbed state",
        )
        perturbed_structure = _WORKER_CONVERSION_SERVICE.pose_to_structure(
            perturbed_pose
        )
        perturbed_structure = _WORKER_CONVERSION_SERVICE.prepare_structure_for_openmm(
            perturbed_structure
        )
        reference_energy = _WORKER_ENERGY_EVALUATOR.evaluate(
            structure=reference_structure,
            restraint_specs=openmm_restraint_specs,
        )
        perturbed_energy = _WORKER_ENERGY_EVALUATOR.evaluate(
            structure=perturbed_structure,
            restraint_specs=openmm_restraint_specs,
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
            restraint_specs,
            job.frame_index,
        )
        if _WORKER_CONFIG.perturbation.original_residue_name is not None:
            ensure_pose_residue_matches_expected(
                prepared_reference_pose,
                _WORKER_CONFIG.perturbation.target_residue_number,
                _WORKER_CONFIG.perturbation.original_residue_name,
                context="Rosetta-relaxed reference state",
            )
        perturbed_pose = _WORKER_PERTURBATION_PROTOCOL.apply_perturbation(
            prepared_reference_pose,
            _WORKER_CONFIG.perturbation.target_residue_number,
            _WORKER_CONFIG.perturbation.target_residue_name,
            restraint_specs,
            job.frame_index,
        )
        ensure_pose_residue_matches_expected(
            perturbed_pose,
            _WORKER_CONFIG.perturbation.target_residue_number,
            _WORKER_CONFIG.perturbation.target_residue_name,
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
        )
        perturbed_energy = _WORKER_ENERGY_EVALUATOR.evaluate(
            pose=perturbed_pose,
            restraint_specs=restraint_specs,
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

    return FrameResult(
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

    def run(self, jobs: Iterable[SnapshotJob]) -> list[FrameResult]:
        """Execute a batch of jobs using the configured parallelism strategy."""
        jobs = list(jobs)
        if not jobs:
            return []

        show_bar = _should_show_progress_bar(self.config, len(jobs))

        if self.config.parallel.workers == 1:
            initialize_worker(self.config, self.atlas, self.assignment_policy)
            results: list[FrameResult] = []
            for job in _maybe_tqdm(
                jobs, total=len(jobs), desc="frames", enabled=show_bar
            ):
                result = safe_process_frame_job(job)
                if self.config.parallel.fail_fast and result.metadata.get("status") == "failed":
                    raise RuntimeError(
                        f"Frame {result.frame_index} failed while --fail-fast was enabled: "
                        f"{result.metadata.get('error_type')}: {result.metadata.get('error_message')}"
                    ) from None
                results.append(result)
            return results

        context = mp.get_context(self.config.parallel.start_method)
        with ProcessPoolExecutor(
            max_workers=self.config.parallel.workers,
            mp_context=context,
            initializer=initialize_worker,
            initargs=(self.config, self.atlas, self.assignment_policy),
        ) as executor:
            futures = [
                executor.submit(safe_process_frame_job, job)
                for job in jobs
            ]
            results_by_future: dict[Future[FrameResult], FrameResult] = {}
            completed_iter = as_completed(futures)
            for future in _maybe_tqdm(
                completed_iter,
                total=len(futures),
                desc="frames",
                enabled=show_bar,
            ):
                result = future.result()
                if self.config.parallel.fail_fast and result.metadata.get("status") == "failed":
                    for pending in futures:
                        pending.cancel()
                    raise RuntimeError(
                        f"Frame {result.frame_index} failed while --fail-fast was enabled: "
                        f"{result.metadata.get('error_type')}: {result.metadata.get('error_message')}"
                    ) from None
                results_by_future[future] = result

        ordered = [results_by_future[future] for future in futures]
        return sorted(ordered, key=lambda item: item.frame_index)
