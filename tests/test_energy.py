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


def test_build_energy_evaluator_openmm_propagates_cpu_threads() -> None:
    evaluator = build_energy_evaluator(
        EnergyConfig(backend="openmm", platform_name="CPU", cpu_threads=4),
        StructureConversionService(pyrosetta_init_options=""),
    )
    assert isinstance(evaluator, OpenMMEnergyEvaluator)
    assert evaluator.platform_name == "CPU"
    assert evaluator.cpu_threads == 4


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


def test_openmm_find_atom_index_falls_back_to_residue_number_without_bridge_metadata() -> None:
    """Unbridged structures may still resolve atoms through residue.number."""
    ev = _minimal_openmm_evaluator()
    struct = Structure()
    struct.add_atom(Atom(name="N", atomic_number=7, mass=14.0), "GLY", 2)
    struct.add_atom(Atom(name="CA", atomic_number=6, mass=12.0), "GLY", 2)
    assert ev._find_atom_index(struct, 2, "CA") == 1


def test_openmm_find_atom_index_prefers_bridge_seqpos_over_pdb_number() -> None:
    """Rosetta bridge seqpos should win when PDB numbering would hit the wrong residue."""
    ev = _minimal_openmm_evaluator()
    struct = Structure()
    struct.add_atom(Atom(name="CA", atomic_number=6, mass=12.0), "ACE", 2)
    struct.add_atom(Atom(name="CA", atomic_number=6, mass=12.0), "GLY", 50)
    setattr(
        struct,
        "_rosetta_bridge_metadata",
        {"residues": [{"seqpos": 1}, {"seqpos": 2}]},
    )
    assert ev._find_atom_index(struct, 2, "CA") == 1
