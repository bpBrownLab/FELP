"""Energy evaluators for FELP."""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from dataclasses import dataclass
from typing import Iterable

from felp.core import DihedralRestraintSpec, EnergyResult
from felp.conversion import StructureConversionService


def _import_openmm():
    import openmm
    import openmm.unit as unit
    from openmm import CustomTorsionForce, LangevinIntegrator
    from openmm.app import ForceField, Modeller, NoCutoff, Simulation

    return openmm, unit, CustomTorsionForce, ForceField, LangevinIntegrator, Modeller, NoCutoff, Simulation


def _import_pyrosetta():
    import pyrosetta
    from pyrosetta.rosetta.core.id import AtomID

    return pyrosetta, AtomID


class EnergyEvaluator(ABC):
    """Common evaluator interface."""

    @abstractmethod
    def evaluate(
        self,
        *,
        structure=None,
        pose=None,
        restraint_specs: Iterable[DihedralRestraintSpec] = (),
    ) -> EnergyResult:
        """Evaluate a structure or pose."""


@dataclass
class OpenMMEnergyEvaluator(EnergyEvaluator):
    """OpenMM evaluator using AMBER XML force fields."""

    protein_forcefield: str
    solvent_forcefield: str
    temperature_kelvin: float
    friction_per_ps: float
    timestep_fs: float
    nonbonded_cutoff_nm: float
    minimization_tolerance_kj_per_mol_nm: float
    minimization_max_iterations: int
    platform_name: str | None = None

    def _find_atom_index(self, structure, residue_number: int, atom_name: str) -> int:
        """Find the 0-based index of an atom in a ParmEd Structure."""
        for atom_index, atom in enumerate(structure.atoms):
            if atom.residue.number == residue_number and atom.name.strip() == atom_name:
                return atom_index
        raise KeyError(
            f"Could not locate atom {atom_name} in residue {residue_number} for OpenMM restraints."
        )

    def _apply_restraints(self, system, structure, restraint_specs):
        """Apply dihedral restraints to the OpenMM System."""
        _, _, CustomTorsionForce, _, _, _, _, _ = _import_openmm()
        force = CustomTorsionForce("0.5*k*(acos(cos(theta-theta0)))^2")
        force.addPerTorsionParameter("theta0")
        force.addPerTorsionParameter("k")

        for spec in restraint_specs:
            atom_indices = [
                self._find_atom_index(structure, residue_number, atom_name)
                for residue_number, atom_name in zip(spec.residue_numbers, spec.atom_names)
            ]
            force.addTorsion(
                atom_indices[0],
                atom_indices[1],
                atom_indices[2],
                atom_indices[3],
                [math.radians(spec.target_angle_degrees), float(spec.force_constant)],
            )

        system.addForce(force)

    def evaluate(
        self,
        *,
        structure=None,
        pose=None,
        restraint_specs: Iterable[DihedralRestraintSpec] = (),
    ) -> EnergyResult:
        """Minimize and evaluate the energy of a ParmEd Structure using OpenMM."""
        if structure is None:
            raise ValueError("OpenMMEnergyEvaluator requires a ParmEd structure.")

        openmm, unit, _, ForceField, LangevinIntegrator, Modeller, NoCutoff, Simulation = _import_openmm()
        modeller = Modeller(structure.topology, structure.positions)
        force_field = ForceField(self.protein_forcefield, self.solvent_forcefield)
        system = force_field.createSystem(
            modeller.topology,
            nonbondedMethod=NoCutoff,
            nonbondedCutoff=self.nonbonded_cutoff_nm * unit.nanometer,
        )

        restraint_specs = list(restraint_specs)
        if restraint_specs:
            self._apply_restraints(system, structure, restraint_specs)

        integrator = LangevinIntegrator(
            self.temperature_kelvin * unit.kelvin,
            self.friction_per_ps / unit.picosecond,
            self.timestep_fs * unit.femtosecond,
        )
        platform = None
        if self.platform_name:
            platform = openmm.Platform.getPlatformByName(self.platform_name)

        if platform is None:
            simulation = Simulation(modeller.topology, system, integrator)
        else:
            simulation = Simulation(modeller.topology, system, integrator, platform)

        simulation.context.setPositions(modeller.positions)
        simulation.minimizeEnergy(
            tolerance=self.minimization_tolerance_kj_per_mol_nm
            * unit.kilojoule_per_mole
            / unit.nanometer,
            maxIterations=self.minimization_max_iterations,
        )
        state = simulation.context.getState(getEnergy=True, getPositions=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        minimized_coordinates = tuple(tuple(float(value) for value in row) for row in positions)
        return EnergyResult(
            backend="openmm",
            energy_kj_mol=float(energy),
            minimized_coordinates_angstrom=minimized_coordinates,
            metadata={"restraint_count": len(restraint_specs)},
        )


@dataclass
class RosettaEnergyEvaluator(EnergyEvaluator):
    """Optional Rosetta scoring backend."""

    conversion_service: StructureConversionService
    score_function_name: str = "ref2015"

    def _build_constraint_set(self, pose, restraint_specs: Iterable[DihedralRestraintSpec]):
        """Convert restraint specs into a PyRosetta ConstraintSet."""
        pyrosetta, AtomID = _import_pyrosetta()
        constraint_set = pyrosetta.rosetta.core.scoring.constraints.ConstraintSet()
        for spec in restraint_specs:
            atom_ids = []
            for residue_number, atom_name in zip(spec.residue_numbers, spec.atom_names):
                atom_index = pose.residue(residue_number).atom_index(atom_name)
                atom_ids.append(AtomID(atom_index, residue_number))
            func = pyrosetta.rosetta.core.scoring.func.CircularHarmonicFunc(
                math.radians(spec.target_angle_degrees),
                max(1.0e-3, 1.0 / math.sqrt(max(spec.force_constant, 1.0e-6))),
            )
            constraint = pyrosetta.rosetta.core.scoring.constraints.DihedralConstraint(
                atom_ids[0],
                atom_ids[1],
                atom_ids[2],
                atom_ids[3],
                func,
            )
            constraint_set.add_constraint(constraint)
        return constraint_set

    def evaluate(
        self,
        *,
        structure=None,
        pose=None,
        restraint_specs: Iterable[DihedralRestraintSpec] = (),
    ) -> EnergyResult:
        """Evaluate the energy of a PyRosetta Pose."""
        self.conversion_service.ensure_pyrosetta_initialized()
        pyrosetta, _ = _import_pyrosetta()
        working_pose = pose.clone() if pose is not None else self.conversion_service.structure_to_pose(structure)
        restraint_specs = list(restraint_specs)
        if restraint_specs:
            working_pose.constraint_set(self._build_constraint_set(working_pose, restraint_specs))
        score_function = pyrosetta.create_score_function(self.score_function_name)
        score_function.set_weight(
            pyrosetta.rosetta.core.scoring.ScoreType.dihedral_constraint, 1.0
        )
        energy = float(score_function(working_pose))
        structure = self.conversion_service.pose_to_structure(working_pose)
        positions = tuple(
            (float(atom.xx), float(atom.xy), float(atom.xz))
            for atom in structure.atoms
        )
        return EnergyResult(
            backend="rosetta",
            energy_kj_mol=energy,
            minimized_coordinates_angstrom=positions,
            metadata={"restraint_count": len(restraint_specs)},
        )


def build_energy_evaluator(
    energy_config,
    conversion_service: StructureConversionService,
) -> EnergyEvaluator:
    """Factory method to build the configured EnergyEvaluator."""
    if energy_config.backend == "openmm":
        return OpenMMEnergyEvaluator(
            protein_forcefield=energy_config.protein_forcefield,
            solvent_forcefield=energy_config.solvent_forcefield,
            temperature_kelvin=energy_config.temperature_kelvin,
            friction_per_ps=energy_config.friction_per_ps,
            timestep_fs=energy_config.timestep_fs,
            nonbonded_cutoff_nm=energy_config.nonbonded_cutoff_nm,
            minimization_tolerance_kj_per_mol_nm=energy_config.minimization_tolerance_kj_per_mol_nm,
            minimization_max_iterations=energy_config.minimization_max_iterations,
            platform_name=energy_config.platform_name,
        )
    return RosettaEnergyEvaluator(
        conversion_service=conversion_service,
        score_function_name=energy_config.rosetta_score_function,
    )
