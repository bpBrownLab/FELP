"""Multiprocessing execution for FELP frame jobs."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from typing import Iterable

from felp.conversion import StructureConversionService
from felp.core import FrameResult, RunConfig, SnapshotJob, default_backbone_cv_specs
from felp.energy import build_energy_evaluator
from felp.perturbation import PyRosettaPerturbationProtocol
from felp.perturbation.validation import (
    ensure_measurement_matches_restraints,
    ensure_perturbation_matches_pose,
    measurement_deviation_degrees,
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
_WORKER_PERTURBATION_CHECKED: bool = False


def _default_assignment_policy(config: RunConfig) -> AssignmentPolicy:
    """Choose the legacy or generalized assignment policy for this run."""

    if config.restraints.cv_specs == default_backbone_cv_specs(
        config.perturbation.target_residue_number
    ):
        return PhiPsiAssignmentPolicy()
    return CVAssignmentPolicy()


def initialize_worker(
    config: RunConfig,
    atlas: MicrostateAtlas,
    assignment_policy: AssignmentPolicy | None = None,
) -> None:
    """Initialize heavy dependencies (PyRosetta, OpenMM) exactly once per worker process."""
    global _WORKER_ATLAS
    global _WORKER_ASSIGNMENT_POLICY
    global _WORKER_PERTURBATION_CHECKED
    global _WORKER_CONFIG
    global _WORKER_CONVERSION_SERVICE
    global _WORKER_ENERGY_EVALUATOR
    global _WORKER_PERTURBATION_PROTOCOL
    global _WORKER_RESTRAINT_PROTOCOL
    global _WORKER_TEMPLATE_STRUCTURE

    _WORKER_CONFIG = config
    _WORKER_ATLAS = atlas
    _WORKER_ASSIGNMENT_POLICY = assignment_policy or _default_assignment_policy(config)
    _WORKER_PERTURBATION_CHECKED = False
    _WORKER_CONVERSION_SERVICE = StructureConversionService(
        pyrosetta_init_options=_build_pyrosetta_options(config),
        radii_set=config.input.radii_set,
    )
    _WORKER_TEMPLATE_STRUCTURE = _WORKER_CONVERSION_SERVICE.load_topology(
        config.input.topology_file
    )
    _WORKER_RESTRAINT_PROTOCOL = BackboneDihedralCVProtocol(
        cv_specs=config.restraints.cv_specs,
        force_constant=config.restraints.force_constant,
    )
    _WORKER_PERTURBATION_PROTOCOL = PyRosettaPerturbationProtocol(
        conversion_service=_WORKER_CONVERSION_SERVICE,
        score_function_name=config.rosetta.score_function,
        relax_iterations=config.rosetta.relax_iterations,
        cartesian=config.rosetta.cartesian,
        output_dir=config.output.output_dir,
        write_pose_pdbs=config.output.write_pose_pdbs,
    )
    _WORKER_ENERGY_EVALUATOR = build_energy_evaluator(
        config.energy,
        _WORKER_CONVERSION_SERVICE,
    )


def _build_pyrosetta_options(config: RunConfig) -> str:
    options = config.rosetta.init_options.strip()
    if config.rosetta.database_path:
        options = f"{options} -database {config.rosetta.database_path}".strip()
    return options


def _measure_structure(structure):
    pose = _WORKER_CONVERSION_SERVICE.structure_to_pose(structure)
    return _WORKER_RESTRAINT_PROTOCOL.measure_pose(pose)


def process_frame_job(job: SnapshotJob) -> FrameResult:
    """Process a single frame: assign microstate, convert, relax, perturb, rescore."""
    global _WORKER_PERTURBATION_CHECKED
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
    if not _WORKER_PERTURBATION_CHECKED:
        ensure_perturbation_matches_pose(reference_pose, _WORKER_CONFIG.perturbation)
        _WORKER_PERTURBATION_CHECKED = True
    reference_measurement = _WORKER_RESTRAINT_PROTOCOL.measure_pose(reference_pose)
    assignment = _WORKER_ASSIGNMENT_POLICY.assign(
        _WORKER_ATLAS,
        job,
        reference_measurement,
    )
    restraint_specs = _WORKER_RESTRAINT_PROTOCOL.build_from_reference_pose(reference_pose)

    if _WORKER_CONFIG.energy.backend == "openmm":
        reference_structure = _WORKER_CONVERSION_SERVICE.pose_to_structure(
            reference_pose
        )
        perturbed_pose = _WORKER_PERTURBATION_PROTOCOL.apply_perturbation(
            reference_pose,
            _WORKER_CONFIG.perturbation.target_residue_number,
            _WORKER_CONFIG.perturbation.target_residue_name,
            restraint_specs,
            job.frame_index,
        )
        perturbed_structure = _WORKER_CONVERSION_SERVICE.pose_to_structure(
            perturbed_pose
        )
        reference_energy = _WORKER_ENERGY_EVALUATOR.evaluate(
            structure=reference_structure,
            restraint_specs=restraint_specs,
        )
        perturbed_energy = _WORKER_ENERGY_EVALUATOR.evaluate(
            structure=perturbed_structure,
            restraint_specs=restraint_specs,
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
        )
        ensure_measurement_matches_restraints(
            perturbed_relaxed_measurement,
            restraint_specs,
            context="OpenMM-minimized perturbed state",
        )
    else:
        prepared_reference_pose = _WORKER_PERTURBATION_PROTOCOL.prepare_reference_pose(
            reference_pose,
            restraint_specs,
            job.frame_index,
        )
        perturbed_pose = _WORKER_PERTURBATION_PROTOCOL.apply_perturbation(
            prepared_reference_pose,
            _WORKER_CONFIG.perturbation.target_residue_number,
            _WORKER_CONFIG.perturbation.target_residue_name,
            restraint_specs,
            job.frame_index,
        )
        reference_relaxed_measurement = _WORKER_RESTRAINT_PROTOCOL.measure_pose(
            prepared_reference_pose
        )
        perturbed_relaxed_measurement = _WORKER_RESTRAINT_PROTOCOL.measure_pose(
            perturbed_pose
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
    reference_deviation = measurement_deviation_degrees(
        reference_relaxed_measurement,
        restraint_specs,
    )
    perturbed_deviation = measurement_deviation_degrees(
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

        if self.config.parallel.workers == 1:
            initialize_worker(self.config, self.atlas, self.assignment_policy)
            return [process_frame_job(job) for job in jobs]

        context = mp.get_context(self.config.parallel.start_method)
        with ProcessPoolExecutor(
            max_workers=self.config.parallel.workers,
            mp_context=context,
            initializer=initialize_worker,
            initargs=(self.config, self.atlas, self.assignment_policy),
        ) as executor:
            results = list(
                executor.map(
                    process_frame_job,
                    jobs,
                    chunksize=self.config.parallel.chunk_size,
                )
            )
        return results
