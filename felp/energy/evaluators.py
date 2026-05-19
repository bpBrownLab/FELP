"""Energy evaluators for FELP."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Iterable

from felp.core import (
    ConfigurationError,
    DihedralRestraintSpec,
    DistanceRestraintSpec,
    EnergyResult,
    MobilityConfig,
)
from felp.conversion import StructureConversionService
from felp.restraints.protocols import find_structure_atom


def resolve_openmm_gpu_platform_name() -> str:
    """Return the first available OpenMM GPU platform name (CUDA, else OpenCL)."""

    import openmm

    for name in ("CUDA", "OpenCL"):
        try:
            openmm.Platform.getPlatformByName(name)
            return name
        except Exception:
            continue
    raise ConfigurationError(
        "OpenMM GPU requested (--gpu) but neither CUDA nor OpenCL platforms are available."
    )


def _import_openmm():
    import openmm
    import openmm.unit as unit
    from openmm import CustomBondForce, CustomTorsionForce, LangevinIntegrator
    from openmm.app import ForceField, Modeller, NoCutoff, Simulation

    return (
        openmm,
        unit,
        CustomBondForce,
        CustomTorsionForce,
        ForceField,
        LangevinIntegrator,
        Modeller,
        NoCutoff,
        Simulation,
    )


def _openmm_import_parts():
    parts = _import_openmm()
    if len(parts) == 8:
        openmm, unit, CustomTorsionForce, ForceField, LangevinIntegrator, Modeller, NoCutoff, Simulation = parts
        CustomBondForce = None
        return (
            openmm,
            unit,
            CustomBondForce,
            CustomTorsionForce,
            ForceField,
            LangevinIntegrator,
            Modeller,
            NoCutoff,
            Simulation,
        )
    return parts


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
        restraint_specs: Iterable[DihedralRestraintSpec | DistanceRestraintSpec] = (),
        collect_energy_components: bool = False,
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
    cpu_threads: int = 1
    mobility: MobilityConfig = MobilityConfig()
    mutation_residue_numbers: tuple[int, ...] = (1,)

    @staticmethod
    def _residue_name_summary(structure) -> str:
        names = sorted(
            {
                str(getattr(residue, "name", "")).strip().upper()
                for residue in getattr(structure, "residues", ())
                if str(getattr(residue, "name", "")).strip()
            }
        )
        return ", ".join(names) if names else "<unknown>"

    def _find_atom_index(self, structure, residue_number: int, atom_name: str) -> int:
        """Find the 0-based index of an atom in a ParmEd Structure.

        Dihedral restraints use **Rosetta pose indices** (1..N). When Rosetta bridge
        metadata is present, use it to resolve the corresponding structure residue.
        Otherwise fall back to pose order and finally ``residue.number``.
        """

        target_atom = find_structure_atom(structure, residue_number, atom_name)
        for atom_index, atom in enumerate(structure.atoms):
            if atom is target_atom:
                return atom_index
        raise KeyError(
            f"Could not locate atom {atom_name} in residue {residue_number} "
            f"(Rosetta metadata, pose order, or residue.number) for OpenMM restraints."
        )

    def _create_system(self, force_field, topology, unit, NoCutoff):
        """Create an OpenMM system while tolerating stripped polymer termini."""

        return force_field.createSystem(
            topology,
            nonbondedMethod=NoCutoff,
            nonbondedCutoff=self.nonbonded_cutoff_nm * unit.nanometer,
            ignoreExternalBonds=True,
        )

    def _apply_restraints(self, system, structure, restraint_specs):
        """Apply CV restraints to the OpenMM System."""
        _, _, CustomBondForce, CustomTorsionForce, _, _, _, _, _ = _openmm_import_parts()
        # Use a smooth periodic restraint to avoid NaN forces at exact target angles
        # on the OpenMM CPU platform. ``acos(cos(theta-theta0))`` is singular at the
        # target by construction for the reference structure, while ``1-cos`` has the
        # same local harmonic curvature without the derivative blow-up.
        force = CustomTorsionForce("k*(1-cos(theta-theta0))")
        force.addPerTorsionParameter("theta0")
        force.addPerTorsionParameter("k")

        torsion_count = 0
        for spec in restraint_specs:
            if not isinstance(spec, DihedralRestraintSpec):
                continue
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
            torsion_count += 1

        if torsion_count:
            system.addForce(force)

        if CustomBondForce is None:
            return
        distance_force = CustomBondForce("0.5*k*(r-r0)^2")
        distance_force.addPerBondParameter("r0")
        distance_force.addPerBondParameter("k")
        distance_count = 0
        for spec in restraint_specs:
            if not isinstance(spec, DistanceRestraintSpec):
                continue
            atom1 = self._find_atom_index(
                structure,
                spec.atom1.residue_number,
                spec.atom1.atom_name,
            )
            atom2 = self._find_atom_index(
                structure,
                spec.atom2.residue_number,
                spec.atom2.atom_name,
            )
            distance_force.addBond(
                atom1,
                atom2,
                [
                    float(spec.target_distance_angstrom) * 0.1,
                    float(spec.force_constant),
                ],
            )
            distance_count += 1
        if distance_count:
            system.addForce(distance_force)

    def _mobile_atom_indices(self, structure) -> set[int]:
        anchors = tuple(int(number) for number in self.mutation_residue_numbers)
        mobile_residues: set[int] = set(anchors)

        def _accumulate_radius(target_residue_number: int) -> None:
            target_residue = find_structure_atom(structure, target_residue_number, "CA").residue
            target_atoms = list(getattr(target_residue, "atoms", ()))
            radius_sq = float(self.mobility.radius_angstrom) ** 2
            for residue in structure.residues:
                for atom in getattr(residue, "atoms", ()):
                    if any(
                        (float(atom.xx) - float(target_atom.xx)) ** 2
                        + (float(atom.xy) - float(target_atom.xy)) ** 2
                        + (float(atom.xz) - float(target_atom.xz)) ** 2
                        <= radius_sq
                        for target_atom in target_atoms
                    ):
                        mobile_residues.add(structure.residues.index(residue) + 1)
                        break

        window = int(self.mobility.sequence_window)
        if self.mobility.mode == "sequence_window":
            mobile_residues.clear()
            for anchor in anchors:
                first_residue = max(1, anchor - window)
                last_residue = min(
                    len(structure.residues),
                    anchor + window,
                )
                mobile_residues.update(range(first_residue, last_residue + 1))

        elif self.mobility.mode == "radius":
            mobile_residues.clear()
            for anchor in anchors:
                _accumulate_radius(anchor)

        mobile_atoms: set[int] = set()
        backbone_names = {"N", "CA", "C", "O", "OXT"}
        for atom_index, atom in enumerate(structure.atoms):
            residue_number = structure.residues.index(atom.residue) + 1
            if residue_number not in mobile_residues:
                continue
            atom_name = str(getattr(atom, "name", "")).strip()
            is_backbone = atom_name in backbone_names
            if (is_backbone and self.mobility.include_backbone) or (
                not is_backbone and self.mobility.include_sidechains
            ):
                mobile_atoms.add(atom_index)
        return mobile_atoms

    def _freeze_immobile_atoms(self, system, structure) -> int:
        mobile_atoms = self._mobile_atom_indices(structure)
        frozen_count = 0
        for atom_index in range(system.getNumParticles()):
            if atom_index not in mobile_atoms:
                system.setParticleMass(atom_index, 0.0)
                frozen_count += 1
        return frozen_count

    def _mobile_atom_summary(self, structure) -> dict:
        mobile_atoms = self._mobile_atom_indices(structure)
        mobile_residues = sorted(
            {
                structure.residues.index(structure.atoms[atom_index].residue) + 1
                for atom_index in mobile_atoms
            }
        )
        return {
            "mobility_mode": self.mobility.mode,
            "target_residue_number": (
                self.mutation_residue_numbers[0]
                if self.mutation_residue_numbers
                else None
            ),
            "mutation_residue_numbers": list(self.mutation_residue_numbers),
            "mobility_radius_angstrom": self.mobility.radius_angstrom,
            "mobility_sequence_window": self.mobility.sequence_window,
            "include_backbone": self.mobility.include_backbone,
            "include_sidechains": self.mobility.include_sidechains,
            "mobile_atom_count": len(mobile_atoms),
            "mobile_residue_numbers": mobile_residues,
            "frozen_atom_scope": (
                "All OpenMM particles outside the mobile atom set are assigned zero mass."
            ),
        }

    def validate_topology(self, structure) -> None:
        """Fail early when the reference topology cannot be templated by OpenMM."""

        if structure is None:
            raise ValueError("OpenMMEnergyEvaluator requires a ParmEd structure.")
        _, unit, _, _, ForceField, _, Modeller, NoCutoff, _ = _openmm_import_parts()
        modeller = Modeller(structure.topology, structure.positions)
        force_field = ForceField(self.protein_forcefield, self.solvent_forcefield)
        try:
            self._create_system(force_field, modeller.topology, unit, NoCutoff)
        except Exception as exc:
            residue_names = self._residue_name_summary(structure)
            raise ConfigurationError(
                "OpenMM could not build a system from the reference topology with "
                f"{self.protein_forcefield} and {self.solvent_forcefield}. "
                f"Residue names present: {residue_names}. For protonation studies, the "
                "topology must use residue names recognized by the selected OpenMM/AMBER "
                "force field (for ff14SB this usually means HID/HIE/HIP/CYM/CYX/ASH/GLH "
                f"as applicable). Original error: {exc}"
            ) from exc

    @staticmethod
    def _force_component_labels(system) -> list[dict[str, int | str]]:
        """Assign OpenMM force groups and return labels for component energies."""

        force_count = system.getNumForces()
        if force_count > 32:
            return []
        labels: list[dict[str, int | str]] = []
        for index in range(force_count):
            force = system.getForce(index)
            force.setForceGroup(index)
            name = ""
            if hasattr(force, "getName"):
                name = str(force.getName() or "")
            labels.append(
                {
                    "force_index": index,
                    "force_group": index,
                    "force_class": type(force).__name__,
                    "name": name,
                    "label": f"{index}:{name or type(force).__name__}",
                }
            )
        return labels

    @staticmethod
    def _collect_energy_components(simulation, force_labels, unit) -> dict:
        """Return per-force-group potential energies in kJ/mol."""

        components: dict[str, float] = {}
        details = []
        for label in force_labels:
            group = int(label["force_group"])
            state = simulation.context.getState(getEnergy=True, groups=1 << group)
            energy = float(
                state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            )
            components[str(label["label"])] = energy
            details.append({**label, "energy_kj_mol": energy})
        return {
            "components_kj_mol": components,
            "components": details,
            "component_sum_kj_mol": float(sum(components.values())),
        }

    def evaluate(
        self,
        *,
        structure=None,
        pose=None,
        restraint_specs: Iterable[DihedralRestraintSpec] = (),
        collect_energy_components: bool = False,
    ) -> EnergyResult:
        """Minimize and evaluate the energy of a ParmEd Structure using OpenMM."""
        if structure is None:
            raise ValueError("OpenMMEnergyEvaluator requires a ParmEd structure.")

        (
            openmm,
            unit,
            _,
            _,
            ForceField,
            LangevinIntegrator,
            Modeller,
            NoCutoff,
            Simulation,
        ) = _openmm_import_parts()
        modeller = Modeller(structure.topology, structure.positions)
        force_field = ForceField(self.protein_forcefield, self.solvent_forcefield)
        system = self._create_system(force_field, modeller.topology, unit, NoCutoff)

        frozen_atom_count = self._freeze_immobile_atoms(system, structure)
        restraint_specs = list(restraint_specs)
        if restraint_specs:
            self._apply_restraints(system, structure, restraint_specs)
        force_labels = (
            self._force_component_labels(system) if collect_energy_components else []
        )

        integrator = LangevinIntegrator(
            self.temperature_kelvin * unit.kelvin,
            self.friction_per_ps / unit.picosecond,
            self.timestep_fs * unit.femtosecond,
        )
        platform = None
        platform_properties: dict[str, str] | None = None
        if self.platform_name:
            platform = openmm.Platform.getPlatformByName(self.platform_name)
            if platform.getName() == "CPU":
                platform_properties = {
                    "Threads": str(max(1, int(self.cpu_threads))),
                }

        if platform is None:
            simulation = Simulation(modeller.topology, system, integrator)
        elif platform_properties is not None:
            simulation = Simulation(
                modeller.topology,
                system,
                integrator,
                platform,
                platform_properties,
            )
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
        metadata = {
            "restraint_count": len(restraint_specs),
            "frozen_atom_count": frozen_atom_count,
            "mobile_dofs": self._mobile_atom_summary(structure),
        }
        if collect_energy_components:
            if force_labels:
                metadata["energy_components"] = self._collect_energy_components(
                    simulation,
                    force_labels,
                    unit,
                )
            else:
                metadata["energy_components"] = {
                    "unavailable_reason": (
                        "OpenMM systems with more than 32 forces cannot be assigned "
                        "unique force groups for component reporting."
                    )
                }
        return EnergyResult(
            backend="openmm",
            energy_kj_mol=float(energy),
            minimized_coordinates_angstrom=minimized_coordinates,
            metadata=metadata,
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
        collect_energy_components: bool = False,
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
    *,
    mobility_config: MobilityConfig | None = None,
    mutation_residue_numbers: tuple[int, ...] = (1,),
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
            cpu_threads=energy_config.cpu_threads,
            mobility=mobility_config or MobilityConfig(),
            mutation_residue_numbers=mutation_residue_numbers,
        )
    return RosettaEnergyEvaluator(
        conversion_service=conversion_service,
        score_function_name=energy_config.rosetta_score_function,
    )
