from __future__ import annotations

from parmed import Structure
from parmed.topologyobjects import Atom

from felp.core import CVSpec
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
