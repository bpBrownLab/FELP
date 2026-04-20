"""PyRosetta perturbation protocols for FELP."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from felp.core import (
    ConversionError,
    DihedralRestraintSpec,
    mutation_target_family,
    normalize_mutation_target_name,
)
from felp.conversion import StructureConversionService
from .validation import ensure_measurement_matches_restraints, target_cv_measurement_from_restraints

_LOCKED_CV_TOLERANCE_DEGREES = 1.0e-3


def _import_pyrosetta():
    import pyrosetta
    from pyrosetta.rosetta.core.id import AtomID
    from pyrosetta.rosetta.protocols.simple_moves import MutateResidue

    return pyrosetta, AtomID, MutateResidue


@dataclass
class PyRosettaPerturbationProtocol:
    """Prepare reference poses and apply residue perturbations."""

    conversion_service: StructureConversionService
    target_residue_number: int
    score_function_name: str = "ref2015"
    perturbation_mode: str = "relax"
    relax_iterations: int = 5
    repack_trials: int = 1
    cartesian: bool = True
    packing_shell_distance_angstrom: float = 6.0
    output_dir: str | None = None
    write_pose_pdbs: bool = False

    def _pose_path(self, frame_index: int, suffix: str) -> str:
        if not self.output_dir:
            raise ValueError("No output directory is configured for pose dumps.")
        pose_dir = Path(self.output_dir) / "poses"
        pose_dir.mkdir(parents=True, exist_ok=True)
        return str(pose_dir / f"frame_{frame_index:05d}_{suffix}.pdb")

    def _observed_residue_name(self, pose, residue_number: int) -> str:
        return normalize_mutation_target_name(
            pose.residue(residue_number).name3().strip().upper()
        )

    def _cv_residue_number(self, spec: DihedralRestraintSpec) -> int:
        if spec.kind == "phi":
            return int(spec.residue_numbers[1])
        return int(spec.residue_numbers[0])

    def _enforce_target_backbone_cvs(
        self,
        pose,
        restraint_specs: Iterable[DihedralRestraintSpec],
    ) -> None:
        restraint_specs = tuple(restraint_specs)
        if not restraint_specs:
            return
        target = target_cv_measurement_from_restraints(restraint_specs)
        for spec in restraint_specs:
            residue_number = self._cv_residue_number(spec)
            target_angle = target.values[spec.label]
            if spec.kind == "phi":
                pose.set_phi(residue_number, target_angle)
            elif spec.kind == "psi":
                pose.set_psi(residue_number, target_angle)
            else:
                pose.set_omega(residue_number, target_angle)

    def _measure_backbone_cvs(
        self,
        pose,
        restraint_specs: Iterable[DihedralRestraintSpec],
    ) -> dict[str, float]:
        measurements: dict[str, float] = {}
        for spec in restraint_specs:
            residue_number = self._cv_residue_number(spec)
            if spec.kind == "phi":
                measurements[spec.label] = float(pose.phi(residue_number))
            elif spec.kind == "psi":
                measurements[spec.label] = float(pose.psi(residue_number))
            else:
                measurements[spec.label] = float(pose.omega(residue_number))
        return measurements

    def _assert_locked_backbone_cvs(
        self,
        pose,
        restraint_specs: Iterable[DihedralRestraintSpec],
        *,
        context: str,
    ) -> None:
        restraint_specs = tuple(restraint_specs)
        if not restraint_specs:
            return
        ensure_measurement_matches_restraints(
            self._measure_backbone_cvs(pose, restraint_specs),
            restraint_specs,
            tolerance_degrees=_LOCKED_CV_TOLERANCE_DEGREES,
            context=context,
        )

    def _build_score_function(self, pyrosetta, *, prefer_cartesian: bool):
        score_function_names = [self.score_function_name]
        if (
            prefer_cartesian
            and self.cartesian
            and not self.score_function_name.endswith("_cart")
        ):
            score_function_names.insert(0, f"{self.score_function_name}_cart")

        score_function = None
        last_error = None
        for score_function_name in score_function_names:
            try:
                score_function = pyrosetta.create_score_function(score_function_name)
                break
            except Exception as exc:  # pragma: no cover - exercised with live PyRosetta
                last_error = exc
        if score_function is None:
            assert last_error is not None
            raise last_error
        return score_function

    def _build_relax_score_function(self, pyrosetta):
        score_function = self._build_score_function(pyrosetta, prefer_cartesian=True)
        if self.cartesian:
            score_type = pyrosetta.rosetta.core.scoring.ScoreType
            cart_bonded_weight = 0.5
            if hasattr(score_function, "get_weight"):
                cart_bonded_weight = max(
                    cart_bonded_weight,
                    float(score_function.get_weight(score_type.cart_bonded)),
                )
            score_function.set_weight(score_type.cart_bonded, cart_bonded_weight)
            if hasattr(score_type, "pro_close"):
                score_function.set_weight(score_type.pro_close, 0.0)
        return score_function

    def _build_packing_score_function(self, pyrosetta):
        return self._build_score_function(pyrosetta, prefer_cartesian=False)

    def _build_packable_selector(self, pyrosetta):
        selectors = pyrosetta.rosetta.core.select.residue_selector
        target_selector = selectors.ResidueIndexSelector(str(self.target_residue_number))
        packable_selector = selectors.NeighborhoodResidueSelector()
        packable_selector.set_focus_selector(target_selector)
        packable_selector.set_distance(self.packing_shell_distance_angstrom)
        packable_selector.set_include_focus_in_subset(True)
        return packable_selector

    def _build_task_factory(self, pyrosetta, packable_selector):
        operations = pyrosetta.rosetta.core.pack.task.operation
        selectors = pyrosetta.rosetta.core.select.residue_selector
        task_factory = pyrosetta.rosetta.core.pack.task.TaskFactory()
        task_factory.push_back(operations.InitializeFromCommandline())
        task_factory.push_back(operations.IncludeCurrent())
        task_factory.push_back(operations.RestrictToRepacking())
        task_factory.push_back(
            operations.OperateOnResidueSubset(
                operations.PreventRepackingRLT(),
                selectors.NotResidueSelector(packable_selector),
            )
        )
        return task_factory

    def _build_movemap(self, pyrosetta, pose, packable_selector):
        movemap = pyrosetta.rosetta.core.kinematics.MoveMap()
        movemap.set_bb(False)
        movemap.set_chi(False)
        if hasattr(movemap, "set_jump"):
            movemap.set_jump(False)

        packable_subset = packable_selector.apply(pose)
        for residue_number in range(1, len(packable_subset) + 1):
            if packable_subset[residue_number]:
                movemap.set_chi(residue_number, True)
        return movemap

    def _prepare_working_pose(
        self,
        pose,
        restraint_specs: Iterable[DihedralRestraintSpec],
        *,
        context: str,
    ):
        working_pose = pose.clone()
        self._enforce_target_backbone_cvs(working_pose, restraint_specs)
        self._assert_locked_backbone_cvs(
            working_pose,
            restraint_specs,
            context=f"{context} setup",
        )
        return working_pose

    def _relax_pose(
        self,
        pose,
        restraint_specs: Iterable[DihedralRestraintSpec],
        frame_index: int,
        suffix: str,
    ):
        self.conversion_service.ensure_pyrosetta_initialized()
        pyrosetta, _, _ = _import_pyrosetta()
        restraint_specs = tuple(restraint_specs)
        working_pose = self._prepare_working_pose(
            pose,
            restraint_specs,
            context="Rosetta FastRelax",
        )

        packable_selector = self._build_packable_selector(pyrosetta)
        task_factory = self._build_task_factory(pyrosetta, packable_selector)
        movemap = self._build_movemap(pyrosetta, working_pose, packable_selector)
        score_function = self._build_relax_score_function(pyrosetta)

        relax = pyrosetta.rosetta.protocols.relax.FastRelax(
            score_function,
            self.relax_iterations,
        )
        relax.set_task_factory(task_factory)
        relax.set_movemap(movemap)
        if hasattr(relax, "set_movemap_disables_packing_of_fixed_chi_positions"):
            relax.set_movemap_disables_packing_of_fixed_chi_positions(True)
        if hasattr(relax, "cartesian"):
            relax.cartesian(self.cartesian)
        if self.cartesian and hasattr(relax, "min_type"):
            relax.min_type("lbfgs_armijo_nonmonotone")
        relax.apply(working_pose)
        self._assert_locked_backbone_cvs(
            working_pose,
            restraint_specs,
            context="Rosetta FastRelax",
        )

        if self.write_pose_pdbs:
            working_pose.dump_pdb(self._pose_path(frame_index, suffix))

        return working_pose

    def _repack_pose(
        self,
        pose,
        restraint_specs: Iterable[DihedralRestraintSpec],
        frame_index: int,
        suffix: str,
    ):
        self.conversion_service.ensure_pyrosetta_initialized()
        pyrosetta, _, _ = _import_pyrosetta()
        restraint_specs = tuple(restraint_specs)
        working_pose = self._prepare_working_pose(
            pose,
            restraint_specs,
            context="Rosetta repack",
        )

        packable_selector = self._build_packable_selector(pyrosetta)
        task_factory = self._build_task_factory(pyrosetta, packable_selector)
        packer_task = task_factory.create_task_and_apply_taskoperations(working_pose)
        score_function = self._build_packing_score_function(pyrosetta)
        repack = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(
            score_function,
            packer_task,
        )
        if self.repack_trials > 1 and hasattr(repack, "nloop"):
            repack.nloop(self.repack_trials)
        repack.apply(working_pose)
        self._assert_locked_backbone_cvs(
            working_pose,
            restraint_specs,
            context="Rosetta repack",
        )

        if self.write_pose_pdbs:
            working_pose.dump_pdb(self._pose_path(frame_index, suffix))

        return working_pose

    def _run_perturbation_protocol(
        self,
        pose,
        restraint_specs: Iterable[DihedralRestraintSpec],
        frame_index: int,
        suffix: str,
    ):
        if self.perturbation_mode == "repack":
            return self._repack_pose(pose, restraint_specs, frame_index, suffix)
        return self._relax_pose(pose, restraint_specs, frame_index, suffix)

    def prepare_reference_pose(
        self,
        pose,
        restraint_specs: Iterable[DihedralRestraintSpec],
        frame_index: int,
    ):
        """Prepare the reference pose without allowing Rosetta CV drift."""
        return self._run_perturbation_protocol(
            pose,
            restraint_specs,
            frame_index,
            f"reference_{self.perturbation_mode}",
        )

    def apply_perturbation(
        self,
        pose,
        target_residue_number: int,
        target_residue_name: str,
        restraint_specs: Iterable[DihedralRestraintSpec],
        frame_index: int,
    ):
        """Mutate the target residue and prepare the perturbed pose."""
        mutated_pose = self.mutate_pose(
            pose,
            target_residue_number,
            target_residue_name,
            restraint_specs,
        )
        return self._run_perturbation_protocol(
            mutated_pose,
            restraint_specs,
            frame_index,
            f"mutated_{target_residue_name.lower()}_{self.perturbation_mode}",
        )

    def mutate_pose(
        self,
        pose,
        target_residue_number: int,
        target_residue_name: str,
        restraint_specs: Iterable[DihedralRestraintSpec] = (),
    ):
        """Mutate the target residue while preserving all configured CV targets."""
        expected = normalize_mutation_target_name(target_residue_name)
        observed = self._observed_residue_name(pose, target_residue_number)
        mutated_pose = pose.clone()
        if observed != expected:
            self.conversion_service.ensure_pyrosetta_initialized()
            _, _, MutateResidue = _import_pyrosetta()
            rosetta_target_name = self.conversion_service.resolve_rosetta_residue_type_name(
                target_residue_name,
                context="mutation target",
            )
            mutate_mover = MutateResidue(target_residue_number, rosetta_target_name)
            mutate_mover.apply(mutated_pose)
        self._enforce_target_backbone_cvs(mutated_pose, restraint_specs)
        self._assert_locked_backbone_cvs(
            mutated_pose,
            restraint_specs,
            context="Rosetta mutation setup",
        )
        return mutated_pose

    def enforce_residue_identity(
        self,
        pose,
        target_residue_number: int,
        target_residue_name: str | None,
        restraint_specs: Iterable[DihedralRestraintSpec] = (),
    ):
        """Return a pose whose target residue matches the requested chemistry."""

        if target_residue_name is None:
            return pose
        expected = normalize_mutation_target_name(target_residue_name)
        observed = self._observed_residue_name(pose, target_residue_number)
        if observed == expected:
            return pose
        if mutation_target_family(observed) != mutation_target_family(expected):
            raise ConversionError(
                f"Cannot enforce residue {expected!r} at Rosetta residue index "
                f"{target_residue_number}: found incompatible input residue {observed!r}. "
                "FELP can normalize protonation/tautomer variants within one residue "
                "family (for example HIS -> HIE/HID), but not replace a different amino acid."
            )
        return self.mutate_pose(
            pose,
            target_residue_number,
            expected,
            restraint_specs,
        )
