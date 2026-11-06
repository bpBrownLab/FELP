from __future__ import annotations

import pytest
from types import SimpleNamespace
from parmed import Structure
from parmed.topologyobjects import Atom

from felp.core import (
    AtomReference,
    CVSpec,
    CustomDihedralCVSpec,
    ConversionError,
    DifferenceDistanceCVSpec,
    DistanceCVSpec,
    DistanceRestraintSpec,
)
from felp.restraints import BackboneDihedralCVProtocol, PhiPsiRestraintProtocol


class _FakePose:
    def __init__(
        self,
        phi: float,
        psi: float,
        omega: float = 180.0,
        total_residue: int = 3,
    ) -> None:
        self._phi = phi
        self._psi = psi
        self._omega = omega
        self._total_residue = total_residue

    def phi(self, residue_number: int) -> float:
        assert residue_number == 2
        return self._phi

    def psi(self, residue_number: int) -> float:
        assert residue_number == 2
        return self._psi

    def omega(self, residue_number: int) -> float:
        assert residue_number == 2
        return self._omega

    def total_residue(self) -> int:
        return self._total_residue


def test_phi_psi_restraint_protocol_uses_reference_angles() -> None:
    pose = _FakePose(phi=-75.0, psi=140.0)
    protocol = PhiPsiRestraintProtocol(target_residue_number=2, force_constant=12.5)

    measurement = protocol.measure_pose(pose)
    restraints = protocol.build_from_reference_pose(pose)

    assert measurement.phi_degrees == -75.0
    assert measurement.psi_degrees == 140.0
    assert [restraint.label for restraint in restraints] == ["phi", "psi"]
    assert [restraint.kind for restraint in restraints] == ["phi", "psi"]
    assert restraints[0].target_angle_degrees == -75.0
    assert restraints[1].target_angle_degrees == 140.0
    assert restraints[0].force_constant == 12.5


def test_backbone_dihedral_protocol_supports_omega() -> None:
    pose = _FakePose(phi=-75.0, psi=140.0, omega=179.0, total_residue=4)
    protocol = BackboneDihedralCVProtocol(
        cv_specs=(
            CVSpec(kind="phi", residue_number=2, name="phi"),
            CVSpec(kind="omega", residue_number=2, name="omega"),
        ),
        force_constant=25.0,
    )

    measurement = protocol.measure_pose(pose)
    restraints = protocol.build_from_reference_pose(pose)

    assert measurement.values == {"phi": -75.0, "omega": 179.0}
    assert [restraint.label for restraint in restraints] == ["phi", "omega"]
    assert [restraint.kind for restraint in restraints] == ["phi", "omega"]
    assert restraints[1].residue_numbers == (2, 2, 3, 3)
    assert restraints[1].atom_names == ("CA", "C", "N", "CA")
    assert restraints[1].target_angle_degrees == 179.0


def test_backbone_dihedral_protocol_measures_structure_with_bridge_seqpos() -> None:
    structure = Structure()
    structure.add_atom(Atom(name="CA", atomic_number=6, mass=12.0), "GLY", 40)
    structure.add_atom(Atom(name="C", atomic_number=6, mass=12.0), "GLY", 40)
    structure.add_atom(Atom(name="N", atomic_number=7, mass=14.0), "ALA", 99)
    structure.add_atom(Atom(name="CA", atomic_number=6, mass=12.0), "ALA", 99)
    coordinates = (
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 1.0, 1.0),
    )
    for atom, xyz in zip(structure.atoms, coordinates):
        atom.xx, atom.xy, atom.xz = xyz
    setattr(
        structure,
        "_rosetta_bridge_metadata",
        {"residues": [{"seqpos": 2}, {"seqpos": 3}]},
    )

    protocol = BackboneDihedralCVProtocol(
        cv_specs=(CVSpec(kind="omega", residue_number=2, name="omega"),),
        force_constant=25.0,
    )

    measurement = protocol.measure_structure(structure)

    assert measurement.values == {"omega": -90.0}


def test_protocol_measures_distance_cv_from_structure() -> None:
    structure = Structure()
    structure.add_atom(Atom(name="CA", atomic_number=6, mass=12.0), "GLY", 1)
    structure.add_atom(Atom(name="CA", atomic_number=6, mass=12.0), "ALA", 2)
    structure.atoms[0].xx, structure.atoms[0].xy, structure.atoms[0].xz = (0.0, 0.0, 0.0)
    structure.atoms[1].xx, structure.atoms[1].xy, structure.atoms[1].xz = (3.0, 4.0, 0.0)
    protocol = BackboneDihedralCVProtocol(
        cv_specs=(),
        distance_cv_specs=(
            DistanceCVSpec(
                name="ca_ca",
                atom1=AtomReference(1, "CA"),
                atom2=AtomReference(2, "CA"),
            ),
        ),
        force_constant=10.0,
    )

    measurement = protocol.measure_structure(structure)

    assert measurement.values == {"ca_ca": 5.0}


def test_protocol_builds_distance_restraint_specs_from_pose() -> None:
    class _XYZ:
        x = 0.0
        y = 0.0
        z = 0.0

    class _Residue:
        def __init__(self, point) -> None:
            self.point = point

        def xyz(self, atom_name: str):
            return self.point

    class _Pose:
        def residue(self, residue_number: int):
            point = _XYZ()
            if residue_number == 2:
                point.x = 0.0
                point.y = 6.0
            return _Residue(point)

    protocol = BackboneDihedralCVProtocol(
        cv_specs=(),
        distance_cv_specs=(
            DistanceCVSpec(
                name="ca_ca",
                atom1=AtomReference(1, "CA"),
                atom2=AtomReference(2, "CA"),
            ),
        ),
        force_constant=10.0,
    )

    restraints = protocol.build_from_reference_pose(_Pose())

    assert isinstance(restraints[0], DistanceRestraintSpec)
    assert restraints[0].target_distance_angstrom == 6.0


def test_protocol_measures_custom_dihedral_cv_from_structure() -> None:
    structure = Structure()
    for index, xyz in enumerate(
        (
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 1.0),
        )
    ):
        structure.add_atom(Atom(name="CA", atomic_number=6, mass=12.0), "GLY", index + 1)
        structure.atoms[-1].xx, structure.atoms[-1].xy, structure.atoms[-1].xz = xyz

    protocol = BackboneDihedralCVProtocol(
        cv_specs=(),
        dihedral_cv_specs=(
            CustomDihedralCVSpec(
                name="aloop",
                atom1=AtomReference(1, "CA"),
                atom2=AtomReference(2, "CA"),
                atom3=AtomReference(3, "CA"),
                atom4=AtomReference(4, "CA"),
            ),
        ),
        force_constant=10.0,
    )

    measurement = protocol.measure_structure(structure)

    assert measurement.values == {"aloop": -90.0}


def _egfr_pseudodihedral_fixture(force_constant=10000.0, with_backbone=False):
    # EGFR's D855/F856/G857/L858 are pose residues 159--162 in the runner.
    numbers = (159, 160, 161, 162)
    points = ((1., 0., 0.), (0., 0., 0.), (0., 1., 0.), (0., 1., 1.))
    xyz = {n: SimpleNamespace(x=x, y=y, z=z) for n, (x, y, z) in zip(numbers, points)}
    pose = SimpleNamespace(
        residue=lambda n: SimpleNamespace(xyz=lambda atom: xyz[n]),
        phi=lambda n: -75.0,
        total_residue=lambda: 200,
    )
    protocol = BackboneDihedralCVProtocol(
        cv_specs=(CVSpec(kind="phi", residue_number=161, name="phi"),) if with_backbone else (),
        dihedral_cv_specs=(CustomDihedralCVSpec(
            name="aloop", **{f"atom{i}": AtomReference(n, "CA") for i, n in enumerate(numbers, 1)},
        ),),
        force_constant=force_constant,
    )
    return protocol, pose, numbers, points


@pytest.mark.parametrize("with_backbone", [False, True])
def test_egfr_pseudodihedral_is_restrained_and_drift_is_rejected(with_backbone):
    from felp.perturbation.validation import ensure_measurement_matches_restraints

    protocol, pose, numbers, _ = _egfr_pseudodihedral_fixture(with_backbone=with_backbone)
    restraints = protocol.build_from_reference_pose(pose)
    assert [s.label for s in restraints] == (["phi", "aloop"] if with_backbone else ["aloop"])
    custom = restraints[-1]
    assert custom.kind == "dihedral"
    assert custom.residue_numbers == numbers
    assert custom.atom_names == ("CA",) * 4
    assert custom.target_angle_degrees == pytest.approx(-90.)
    assert custom.force_constant == 10000.
    measured = protocol.measure_pose(pose).values
    ensure_measurement_matches_restraints(measured, restraints)
    with pytest.raises(ConversionError, match="aloop"):
        ensure_measurement_matches_restraints({**measured, "aloop": -80.}, restraints)


def test_egfr_built_pseudodihedral_generates_openmm_restoring_energy():
    import openmm
    from openmm import unit
    from felp.conversion import StructureConversionService
    from felp.core import EnergyConfig
    from felp.energy import build_energy_evaluator

    protocol, pose, numbers, points = _egfr_pseudodihedral_fixture()
    structure = Structure()
    for number in numbers:
        structure.add_atom(Atom(name="CA", atomic_number=6, mass=12.), "GLY", number)
    system = openmm.System()
    for _ in numbers:
        system.addParticle(12.)
    evaluator = build_energy_evaluator(
        EnergyConfig(), StructureConversionService(pyrosetta_init_options=""),
    )
    evaluator._apply_restraints(system, structure, protocol.build_from_reference_pose(pose))
    assert system.getNumForces() == 1
    integrator = openmm.VerletIntegrator(0.001)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    try:
        context.setPositions(points * unit.angstrom)
        energy = context.getState(getEnergy=True).getPotentialEnergy()
        assert energy.value_in_unit(unit.kilojoule_per_mole) == pytest.approx(0., abs=1e-8)
        # Rotate the fourth CA by 90 degrees about the middle CA--CA axis.
        moved = (*points[:3], (1., 1., 0.))
        context.setPositions(moved * unit.angstrom)
        energy = context.getState(getEnergy=True).getPotentialEnergy()
        assert energy.value_in_unit(unit.kilojoule_per_mole) == pytest.approx(10000.)
    finally:
        del context, integrator


@pytest.mark.integration
def test_egfr_built_pseudodihedral_generates_rosetta_restoring_energy():
    pyrosetta = pytest.importorskip("pyrosetta")
    from felp.conversion import StructureConversionService
    from felp.perturbation.protocols import PyRosettaPerturbationProtocol

    from felp.core import RosettaConfig

    service = StructureConversionService(pyrosetta_init_options=RosettaConfig().init_options)
    service.ensure_pyrosetta_initialized()
    pose = pyrosetta.pose_from_sequence("DFGL")
    pose.set_phi(3, -75.)
    pose.set_psi(3, 130.)
    protocol = BackboneDihedralCVProtocol(
        cv_specs=(), force_constant=10000.,
        dihedral_cv_specs=(CustomDihedralCVSpec(
            name="aloop", **{f"atom{i}": AtomReference(i, "CA") for i in range(1, 5)},
        ),),
    )
    restraints = protocol.build_from_reference_pose(pose)
    assert len(restraints) == 1
    perturbation = PyRosettaPerturbationProtocol(service, mutation_residue_numbers=(4,))
    score = pyrosetta.ScoreFunction()
    score.set_weight(pyrosetta.rosetta.core.scoring.dihedral_constraint, 1.)
    # The same reference target must constrain both endpoint branches.
    for branch in (pose.clone(), pose.clone()):
        perturbation._add_dihedral_constraints(pyrosetta, branch, restraints)
        assert score(branch) == pytest.approx(0., abs=1e-7)
        branch.set_phi(3, -55.)
        assert score(branch) > 1.


def test_protocol_measures_difference_distance_cv_from_structure() -> None:
    structure = Structure()
    structure.add_atom(Atom(name="NZ", atomic_number=7, mass=14.0), "LYS", 1)
    structure.add_atom(Atom(name="OE1", atomic_number=8, mass=16.0), "GLU", 2)
    structure.add_atom(Atom(name="NZ", atomic_number=7, mass=14.0), "LYS", 3)
    structure.atoms[0].xx, structure.atoms[0].xy, structure.atoms[0].xz = (0.0, 0.0, 0.0)
    structure.atoms[1].xx, structure.atoms[1].xy, structure.atoms[1].xz = (3.0, 4.0, 0.0)
    structure.atoms[2].xx, structure.atoms[2].xy, structure.atoms[2].xz = (6.0, 0.0, 0.0)

    protocol = BackboneDihedralCVProtocol(
        cv_specs=(),
        difference_distance_cv_specs=(
            DifferenceDistanceCVSpec(
                name="ac_diff",
                positive_atom1=AtomReference(3, "NZ"),
                positive_atom2=AtomReference(2, "OE1"),
                negative_atom1=AtomReference(1, "NZ"),
                negative_atom2=AtomReference(2, "OE1"),
            ),
        ),
        force_constant=10.0,
    )

    measurement = protocol.measure_structure(structure)

    assert measurement.values["ac_diff"] == pytest.approx(0.0)
    assert measurement.values["ac_diff__positive"] == pytest.approx(5.0)
    assert measurement.values["ac_diff__negative"] == pytest.approx(5.0)

    from felp.core import DistanceRestraintSpec
    from felp.perturbation.validation import ensure_measurement_matches_restraints

    # Restraint labels for difference-distance components must be present in
    # the measurement so post-minimize validation does not mark every frame failed.
    restraints = (
        DistanceRestraintSpec(
            label="ac_diff__positive",
            atom1=AtomReference(3, "NZ"),
            atom2=AtomReference(2, "OE1"),
            target_distance_angstrom=5.0,
            force_constant=10.0,
        ),
        DistanceRestraintSpec(
            label="ac_diff__negative",
            atom1=AtomReference(1, "NZ"),
            atom2=AtomReference(2, "OE1"),
            target_distance_angstrom=5.0,
            force_constant=10.0,
        ),
    )
    ensure_measurement_matches_restraints(
        measurement,
        restraints,
        distance_tolerance_angstrom=0.05,
    )
