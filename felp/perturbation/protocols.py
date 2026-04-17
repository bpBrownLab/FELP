"""PyRosetta perturbation protocols for FELP."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

from felp.core import DihedralRestraintSpec
from felp.conversion import StructureConversionService
from .validation import target_cv_measurement_from_restraints


def _import_pyrosetta():
    import pyrosetta
    from pyrosetta.rosetta.core.id import AtomID
    from pyrosetta.rosetta.protocols.simple_moves import MutateResidue

    return pyrosetta, AtomID, MutateResidue


@dataclass
class PyRosettaPerturbationProtocol:
    """Prepare reference poses and apply residue perturbations."""

    conversion_service: StructureConversionService
    score_function_name: str = "ref2015"
    relax_iterations: int = 5
    cartesian: bool = True
    output_dir: str | None = None
    write_pose_pdbs: bool = False

    def _pose_path(self, frame_index: int, suffix: str) -> str:
        if not self.output_dir:
            raise ValueError("No output directory is configured for pose dumps.")
        pose_dir = Path(self.output_dir) / "poses"
        pose_dir.mkdir(parents=True, exist_ok=True)
        return str(pose_dir / f"frame_{frame_index:05d}_{suffix}.pdb")

    def _build_constraint_set(self, pose, restraint_specs: Iterable[DihedralRestraintSpec]):
        pyrosetta, AtomID, _ = _import_pyrosetta()
        constraint_set = pyrosetta.rosetta.core.scoring.constraints.ConstraintSet()

        for spec in restraint_specs:
            atom_ids = []
            for residue_number, atom_name in zip(spec.residue_numbers, spec.atom_names):
                atom_index = pose.residue(residue_number).atom_index(atom_name)
                atom_ids.append(AtomID(atom_index, residue_number))
            circular_harmonic = pyrosetta.rosetta.core.scoring.func.CircularHarmonicFunc(
                math.radians(spec.target_angle_degrees),
                max(1.0e-3, 1.0 / math.sqrt(max(spec.force_constant, 1.0e-6))),
            )
            constraint = pyrosetta.rosetta.core.scoring.constraints.DihedralConstraint(
                atom_ids[0],
                atom_ids[1],
                atom_ids[2],
                atom_ids[3],
                circular_harmonic,
            )
            constraint_set.add_constraint(constraint)

        return constraint_set

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

    def _build_relax_score_function(self, pyrosetta):
        score_function_names = [self.score_function_name]
        if self.cartesian and not self.score_function_name.endswith("_cart"):
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

        score_type = pyrosetta.rosetta.core.scoring.ScoreType
        score_function.set_weight(score_type.dihedral_constraint, 1.0)
        if self.cartesian:
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

        working_pose = pose.clone()
        working_pose.constraint_set(self._build_constraint_set(working_pose, restraint_specs))

        score_function = self._build_relax_score_function(pyrosetta)

        relax = pyrosetta.rosetta.protocols.relax.FastRelax(
            score_function,
            self.relax_iterations,
        )
        if hasattr(relax, "cartesian"):
            relax.cartesian(self.cartesian)
        if self.cartesian and hasattr(relax, "min_type"):
            relax.min_type("lbfgs_armijo_nonmonotone")
        relax.apply(working_pose)
        self._enforce_target_backbone_cvs(working_pose, restraint_specs)

        if self.write_pose_pdbs:
            working_pose.dump_pdb(self._pose_path(frame_index, suffix))

        return working_pose

    def prepare_reference_pose(
        self,
        pose,
        restraint_specs: Iterable[DihedralRestraintSpec],
        frame_index: int,
    ):
        """Relax the reference pose under the specified restraints."""
        return self._relax_pose(pose, restraint_specs, frame_index, "reference_relaxed")

    def apply_perturbation(
        self,
        pose,
        target_residue_number: int,
        target_residue_name: str,
        restraint_specs: Iterable[DihedralRestraintSpec],
        frame_index: int,
    ):
        """Mutate the target residue and relax the perturbed pose under restraints."""
        mutated_pose = self.mutate_pose(
            pose,
            target_residue_number,
            target_residue_name,
            restraint_specs,
        )
        return self._relax_pose(
            mutated_pose,
            restraint_specs,
            frame_index,
            f"mutated_{target_residue_name.lower()}",
        )

    def mutate_pose(
        self,
        pose,
        target_residue_number: int,
        target_residue_name: str,
        restraint_specs: Iterable[DihedralRestraintSpec] = (),
    ):
        """Mutate the target residue while preserving all configured CV targets."""
        self.conversion_service.ensure_pyrosetta_initialized()
        _, _, MutateResidue = _import_pyrosetta()

        mutated_pose = pose.clone()
        mutate_mover = MutateResidue(target_residue_number, target_residue_name)
        mutate_mover.apply(mutated_pose)
        self._enforce_target_backbone_cvs(mutated_pose, restraint_specs)
        return mutated_pose
