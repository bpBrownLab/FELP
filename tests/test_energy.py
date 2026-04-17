from __future__ import annotations

from parmed import Structure
from parmed.topologyobjects import Atom

from felp.conversion import StructureConversionService
from felp.core import EnergyConfig
from felp.energy import OpenMMEnergyEvaluator, RosettaEnergyEvaluator, build_energy_evaluator


def test_build_energy_evaluator_openmm() -> None:
    evaluator = build_energy_evaluator(
        EnergyConfig(backend="openmm"),
        StructureConversionService(pyrosetta_init_options=""),
    )
    assert isinstance(evaluator, OpenMMEnergyEvaluator)


def test_build_energy_evaluator_rosetta() -> None:
    evaluator = build_energy_evaluator(
        EnergyConfig(backend="rosetta"),
        StructureConversionService(pyrosetta_init_options=""),
    )
    assert isinstance(evaluator, RosettaEnergyEvaluator)


def _minimal_openmm_evaluator() -> OpenMMEnergyEvaluator:
    return OpenMMEnergyEvaluator(
        protein_forcefield="amber/protein.ff14SBonlysc.xml",
        solvent_forcefield="implicit/gbn2.xml",
        temperature_kelvin=300.0,
        friction_per_ps=1.0,
        timestep_fs=2.0,
        nonbonded_cutoff_nm=1.0,
        minimization_tolerance_kj_per_mol_nm=1.0,
        minimization_max_iterations=0,
    )


def test_openmm_find_atom_index_matches_pdb_number_first() -> None:
    """When ParmEd residue.number equals the restraint index, use it."""
    ev = _minimal_openmm_evaluator()
    struct = Structure()
    struct.add_atom(Atom(name="N", atomic_number=7, mass=14.0), "GLY", 2)
    struct.add_atom(Atom(name="CA", atomic_number=6, mass=12.0), "GLY", 2)
    assert ev._find_atom_index(struct, 2, "CA") == 1


def test_openmm_find_atom_index_falls_back_to_pose_order() -> None:
    """Pose index may differ from PDB residue.number after Rosetta bridge."""
    ev = _minimal_openmm_evaluator()
    struct = Structure()
    struct.add_atom(Atom(name="C", atomic_number=6, mass=12.0), "ACE", 1)
    struct.add_atom(Atom(name="CA", atomic_number=6, mass=12.0), "GLY", 50)
    assert ev._find_atom_index(struct, 2, "CA") == 1
