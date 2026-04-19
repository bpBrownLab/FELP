from __future__ import annotations

from types import SimpleNamespace

import pytest

from felp.core import ConversionError, DihedralRestraintSpec
from felp.perturbation.protocols import PyRosettaPerturbationProtocol


class _DummyConversionService:
    def ensure_pyrosetta_initialized(self) -> None:
        return None

    def resolve_rosetta_residue_type_name(self, residue_name: str, *, context: str) -> str:
        return {"ASH": "ASP_P2", "GLH": "GLU_P2", "HIE": "HIS_E", "HID": "HIS_D", "HIP": "HIS_P"}.get(
            residue_name,
            residue_name,
        )


class _DummyPose:
    def __init__(
        self,
        phi: float = -80.0,
        psi: float = 150.0,
        residue_name: str = "GLY",
    ) -> None:
        self.constraint_value = None
        self._phi = float(phi)
        self._psi = float(psi)
        self._residue_name = str(residue_name)

    def clone(self):
        clone = _DummyPose(self._phi, self._psi, self._residue_name)
        clone.constraint_value = self.constraint_value
        return clone

    def constraint_set(self, value) -> None:
        self.constraint_value = value

    def phi(self, residue_number: int) -> float:
        assert residue_number == 2
        return self._phi

    def psi(self, residue_number: int) -> float:
        assert residue_number == 2
        return self._psi

    def set_phi(self, residue_number: int, value: float) -> None:
        assert residue_number == 2
        self._phi = float(value)

    def set_psi(self, residue_number: int, value: float) -> None:
        assert residue_number == 2
        self._psi = float(value)

    def residue(self, residue_number: int):
        assert residue_number == 2
        return SimpleNamespace(name3=lambda: self._residue_name)


class _DummyScoreFunction:
    def __init__(self, name: str) -> None:
        self.name = name
        self.weights: dict[str, float] = {}

    def set_weight(self, score_type: str, value: float) -> None:
        self.weights[score_type] = float(value)

    def get_weight(self, score_type: str) -> float:
        return self.weights.get(score_type, 0.0)


def _restraint_specs() -> list[DihedralRestraintSpec]:
    return [
        DihedralRestraintSpec(
            label="phi",
            kind="phi",
            residue_numbers=(1, 2, 2, 2),
            atom_names=("C", "N", "CA", "C"),
            target_angle_degrees=-60.0,
            force_constant=10000.0,
        ),
        DihedralRestraintSpec(
            label="psi",
            kind="psi",
            residue_numbers=(2, 2, 2, 3),
            atom_names=("N", "CA", "C", "N"),
            target_angle_degrees=140.0,
            force_constant=10000.0,
        ),
    ]


def test_relax_pose_uses_cartesian_ready_scorefunction(monkeypatch) -> None:
    create_calls: list[str] = []
    relax_calls: dict[str, object] = {}

    def create_score_function(name: str):
        create_calls.append(name)
        if name == "ref2015_cart":
            score_function = _DummyScoreFunction(name)
            score_function.set_weight("cart_bonded", 0.5)
            return score_function
        raise RuntimeError(name)

    class _FakeFastRelax:
        def __init__(self, score_function, iterations: int) -> None:
            relax_calls["score_function"] = score_function
            relax_calls["iterations"] = iterations

        def cartesian(self, enabled: bool) -> None:
            relax_calls["cartesian"] = enabled

        def min_type(self, value: str) -> None:
            relax_calls["min_type"] = value

        def apply(self, pose) -> None:
            pose.set_phi(2, -10.0)
            pose.set_psi(2, 25.0)
            relax_calls["pose"] = pose

    fake_pyrosetta = SimpleNamespace(
        create_score_function=create_score_function,
        rosetta=SimpleNamespace(
            core=SimpleNamespace(
                scoring=SimpleNamespace(
                    ScoreType=SimpleNamespace(
                        dihedral_constraint="dihedral_constraint",
                        cart_bonded="cart_bonded",
                        pro_close="pro_close",
                    )
                )
            ),
            protocols=SimpleNamespace(
                relax=SimpleNamespace(
                    FastRelax=_FakeFastRelax,
                )
            ),
        ),
    )

    monkeypatch.setattr(
        "felp.perturbation.protocols._import_pyrosetta",
        lambda: (fake_pyrosetta, object(), object()),
    )
    monkeypatch.setattr(
        PyRosettaPerturbationProtocol,
        "_build_constraint_set",
        lambda self, pose, restraint_specs: "constraints",
    )

    protocol = PyRosettaPerturbationProtocol(
        conversion_service=_DummyConversionService(),
        score_function_name="ref2015",
        relax_iterations=3,
        cartesian=True,
    )
    result_pose = protocol._relax_pose(_DummyPose(), _restraint_specs(), 0, "reference")

    assert create_calls == ["ref2015_cart"]
    assert relax_calls["iterations"] == 3
    assert relax_calls["cartesian"] is True
    assert relax_calls["min_type"] == "lbfgs_armijo_nonmonotone"
    assert relax_calls["score_function"].weights["dihedral_constraint"] == 1.0
    assert relax_calls["score_function"].weights["cart_bonded"] == 0.5
    assert relax_calls["score_function"].weights["pro_close"] == 0.0
    assert isinstance(result_pose, _DummyPose)
    assert result_pose.constraint_value == "constraints"
    assert result_pose.phi(2) == -60.0
    assert result_pose.psi(2) == 140.0


def test_enforce_residue_identity_mutates_within_same_family(monkeypatch) -> None:
    mutate_calls: list[tuple[int, str]] = []

    class _FakeMutateResidue:
        def __init__(self, residue_number: int, residue_name: str) -> None:
            self.residue_number = residue_number
            self.residue_name = residue_name

        def apply(self, pose) -> None:
            mutate_calls.append((self.residue_number, self.residue_name))
            pose._residue_name = self.residue_name

    monkeypatch.setattr(
        "felp.perturbation.protocols._import_pyrosetta",
        lambda: (SimpleNamespace(), object(), _FakeMutateResidue),
    )

    protocol = PyRosettaPerturbationProtocol(
        conversion_service=_DummyConversionService(),
    )
    result_pose = protocol.enforce_residue_identity(_DummyPose(residue_name="HIS"), 2, "HIE")

    assert mutate_calls == [(2, "HIS_E")]
    assert result_pose.residue(2).name3() == "HIS_E"


def test_enforce_residue_identity_uses_rosetta_protonated_acid_name(monkeypatch) -> None:
    mutate_calls: list[tuple[int, str]] = []

    class _FakeMutateResidue:
        def __init__(self, residue_number: int, residue_name: str) -> None:
            self.residue_number = residue_number
            self.residue_name = residue_name

        def apply(self, pose) -> None:
            mutate_calls.append((self.residue_number, self.residue_name))
            pose._residue_name = self.residue_name

    monkeypatch.setattr(
        "felp.perturbation.protocols._import_pyrosetta",
        lambda: (SimpleNamespace(), object(), _FakeMutateResidue),
    )

    protocol = PyRosettaPerturbationProtocol(
        conversion_service=_DummyConversionService(),
    )
    result_pose = protocol.enforce_residue_identity(_DummyPose(residue_name="ASP"), 2, "ASH")

    assert mutate_calls == [(2, "ASP_P2")]
    assert result_pose.residue(2).name3() == "ASP_P2"


def test_enforce_residue_identity_rejects_incompatible_family(monkeypatch) -> None:
    protocol = PyRosettaPerturbationProtocol(
        conversion_service=_DummyConversionService(),
    )

    with pytest.raises(ConversionError, match="incompatible input residue"):
        protocol.enforce_residue_identity(_DummyPose(residue_name="GLU"), 2, "HIE")
