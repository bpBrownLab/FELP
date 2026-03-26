"""PyRosetta perturbation protocols for FELP."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

from felp.core import DihedralRestraintSpec
from felp.conversion import StructureConversionService


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

    def _relax_pose(
        self,
        pose,
        restraint_specs: Iterable[DihedralRestraintSpec],
        frame_index: int,
        suffix: str,
    ):
        self.conversion_service.ensure_pyrosetta_initialized()
        pyrosetta, _, _ = _import_pyrosetta()

        working_pose = pose.clone()
        working_pose.constraint_set(self._build_constraint_set(working_pose, restraint_specs))

        score_function = pyrosetta.create_score_function(self.score_function_name)
        score_function.set_weight(
            pyrosetta.rosetta.core.scoring.ScoreType.dihedral_constraint, 1.0
        )
        if self.cartesian:
            score_function.set_weight(
                pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded, 0.5
            )

        relax = pyrosetta.rosetta.protocols.relax.FastRelax(
            score_function,
            self.relax_iterations,
        )
        if hasattr(relax, "cartesian"):
            relax.cartesian(self.cartesian)
        relax.apply(working_pose)

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
        self.conversion_service.ensure_pyrosetta_initialized()
        _, _, MutateResidue = _import_pyrosetta()

        mutated_pose = pose.clone()
        reference_phi = mutated_pose.phi(target_residue_number)
        reference_psi = mutated_pose.psi(target_residue_number)

        mutate_mover = MutateResidue(target_residue_number, target_residue_name)
        mutate_mover.apply(mutated_pose)
        mutated_pose.set_phi(target_residue_number, reference_phi)
        mutated_pose.set_psi(target_residue_number, reference_psi)

        return self._relax_pose(
            mutated_pose,
            restraint_specs,
            frame_index,
            f"mutated_{target_residue_name.lower()}",
        )
