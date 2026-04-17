from __future__ import annotations

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
