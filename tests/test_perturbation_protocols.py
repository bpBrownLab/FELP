from __future__ import annotations

from types import SimpleNamespace

from felp.perturbation.protocols import PyRosettaPerturbationProtocol


class _DummyConversionService:
    def ensure_pyrosetta_initialized(self) -> None:
        return None


class _DummyPose:
    def __init__(self) -> None:
        self.constraint_value = None

    def clone(self):
        return _DummyPose()

    def constraint_set(self, value) -> None:
        self.constraint_value = value


class _DummyScoreFunction:
    def __init__(self, name: str) -> None:
        self.name = name
        self.weights: dict[str, float] = {}

    def set_weight(self, score_type: str, value: float) -> None:
        self.weights[score_type] = float(value)

    def get_weight(self, score_type: str) -> float:
        return self.weights.get(score_type, 0.0)


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
    result_pose = protocol._relax_pose(_DummyPose(), [], 0, "reference")

    assert create_calls == ["ref2015_cart"]
    assert relax_calls["iterations"] == 3
    assert relax_calls["cartesian"] is True
    assert relax_calls["min_type"] == "lbfgs_armijo_nonmonotone"
    assert relax_calls["score_function"].weights["dihedral_constraint"] == 1.0
    assert relax_calls["score_function"].weights["cart_bonded"] == 0.5
    assert relax_calls["score_function"].weights["pro_close"] == 0.0
    assert isinstance(result_pose, _DummyPose)
    assert result_pose.constraint_value == "constraints"
