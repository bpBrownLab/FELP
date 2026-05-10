from __future__ import annotations

from parmed import Structure
from parmed.topologyobjects import Atom
import pytest

from felp.conversion import StructureConversionService
from felp.core import ConfigurationError, DihedralRestraintSpec, EnergyConfig
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


class _FakeModeller:
    def __init__(self, topology, positions) -> None:
        self.topology = topology
        self.positions = positions


class _FakeForceField:
    def __init__(self, *args) -> None:
        self.args = args
        self.create_system_kwargs = None

    def createSystem(self, topology, **kwargs):
        self.create_system_kwargs = kwargs
        return object()


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


def test_openmm_validate_topology_reports_residue_names_on_template_failure(
    monkeypatch,
) -> None:
    ev = _minimal_openmm_evaluator()

    class _FailingForceField(_FakeForceField):
        def createSystem(self, topology, **kwargs):
            raise ValueError("No template found for residue 1")

    structure = type(
        "FakeStructure",
        (),
        {
            "topology": object(),
            "positions": object(),
            "residues": [type("Residue", (), {"name": "HIS"})(), type("Residue", (), {"name": "GLH"})()],
        },
    )()
    monkeypatch.setattr(
        "felp.energy.evaluators._import_openmm",
        lambda: (None, type("Unit", (), {"nanometer": 1.0})(), None, _FailingForceField, None, _FakeModeller, object(), None),
    )

    with pytest.raises(ConfigurationError, match="GLH, HIS"):
        ev.validate_topology(structure)


def test_openmm_create_system_ignores_external_bonds_for_stripped_termini() -> None:
    ev = _minimal_openmm_evaluator()
    force_field = _FakeForceField()
    unit = type("Unit", (), {"nanometer": 1.0})()

    ev._create_system(force_field, object(), unit, object())

    assert force_field.create_system_kwargs["ignoreExternalBonds"] is True


class _FakeCustomTorsionForce:
    def __init__(self, energy_function: str) -> None:
        self.energy_function = energy_function
        self.parameters: list[str] = []
        self.torsions: list[tuple[int, int, int, int, list[float]]] = []

    def addPerTorsionParameter(self, name: str) -> None:
        self.parameters.append(name)

    def addTorsion(self, a: int, b: int, c: int, d: int, values: list[float]) -> None:
        self.torsions.append((a, b, c, d, values))


class _FakeSystem:
    def __init__(self) -> None:
        self.forces: list[_FakeCustomTorsionForce] = []

    def addForce(self, force: _FakeCustomTorsionForce) -> None:
        self.forces.append(force)


def test_openmm_restraints_use_smooth_periodic_torsion_kernel(monkeypatch) -> None:
    ev = _minimal_openmm_evaluator()
    struct = Structure()
    struct.add_atom(Atom(name="C", atomic_number=6, mass=12.0), "ACE", 1)
    struct.add_atom(Atom(name="N", atomic_number=7, mass=14.0), "GLY", 2)
    struct.add_atom(Atom(name="CA", atomic_number=6, mass=12.0), "GLY", 2)
    struct.add_atom(Atom(name="C", atomic_number=6, mass=12.0), "GLY", 2)
    struct.add_atom(Atom(name="N", atomic_number=7, mass=14.0), "ALA", 3)

    fake_system = _FakeSystem()

    monkeypatch.setattr(
        'felp.energy.evaluators._import_openmm',
        lambda: (None, None, _FakeCustomTorsionForce, None, None, None, None, None),
    )

    ev._apply_restraints(
        fake_system,
        struct,
        [
            DihedralRestraintSpec(
                label="psi",
                kind="psi",
                residue_numbers=(2, 2, 2, 3),
                atom_names=("N", "CA", "C", "N"),
                target_angle_degrees=140.0,
                force_constant=25.0,
            )
        ],
    )

    assert len(fake_system.forces) == 1
    force = fake_system.forces[0]
    assert force.energy_function == "k*(1-cos(theta-theta0))"
    assert force.parameters == ["theta0", "k"]
    assert force.torsions[0][:4] == (1, 2, 3, 4)
