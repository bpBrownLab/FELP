from __future__ import annotations

from felp.restraints import PhiPsiRestraintProtocol


class _FakePose:
    def __init__(self, phi: float, psi: float, total_residue: int = 3) -> None:
        self._phi = phi
        self._psi = psi
        self._total_residue = total_residue

    def phi(self, residue_number: int) -> float:
        assert residue_number == 2
        return self._phi

    def psi(self, residue_number: int) -> float:
        assert residue_number == 2
        return self._psi

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
    assert restraints[0].target_angle_degrees == -75.0
    assert restraints[1].target_angle_degrees == 140.0
    assert restraints[0].force_constant == 12.5
