from __future__ import annotations
from types import SimpleNamespace

import pytest

from felp.core import (
    AtomReference,
    ConversionError,
    DihedralRestraintSpec,
    DistanceRestraintSpec,
    MobilityConfig,
    MutationSpec,
)
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
        omega: float = 180.0,
        residue_name: str = "GLY",
    ) -> None:
        self._phi = float(phi)
        self._psi = float(psi)
        self._omega = float(omega)
        self._residue_name = str(residue_name)
        self.constraints: list[object] = []

    def clone(self):
        return _DummyPose(self._phi, self._psi, self._omega, self._residue_name)

    def phi(self, residue_number: int) -> float:
        assert residue_number == 2
        return self._phi

    def psi(self, residue_number: int) -> float:
        assert residue_number == 2
        return self._psi

    def omega(self, residue_number: int) -> float:
        assert residue_number == 2
        return self._omega

    def set_phi(self, residue_number: int, value: float) -> None:
        assert residue_number == 2
        self._phi = float(value)

    def set_psi(self, residue_number: int, value: float) -> None:
        assert residue_number == 2
        self._psi = float(value)

    def set_omega(self, residue_number: int, value: float) -> None:
        assert residue_number == 2
        self._omega = float(value)

    def residue(self, residue_number: int):
        assert residue_number in {1, 2, 3}
        return SimpleNamespace(
            name3=lambda: self._residue_name,
            atom_index=lambda atom_name: {"C": 1, "N": 2, "CA": 3}.get(atom_name, 4),
        )

    def add_constraint(self, constraint) -> None:
        self.constraints.append(constraint)

    def dump_pdb(self, path: str) -> None:
        self.dumped_path = path


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


def test_relax_pose_uses_locked_movemap_and_task_factory(monkeypatch) -> None:
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

        def set_task_factory(self, task_factory) -> None:
            relax_calls["task_factory"] = task_factory

        def set_movemap(self, movemap) -> None:
            relax_calls["movemap"] = movemap

        def set_movemap_disables_packing_of_fixed_chi_positions(
            self,
            enabled: bool,
        ) -> None:
            relax_calls["disable_fixed_chi_packing"] = enabled

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
                id=SimpleNamespace(AtomID=lambda atom_index, res: (atom_index, res)),
                scoring=SimpleNamespace(
                    ScoreType=SimpleNamespace(
                        dihedral_constraint="dihedral_constraint",
                        cart_bonded="cart_bonded",
                        pro_close="pro_close",
                    ),
                    constraints=SimpleNamespace(
                        DihedralConstraint=lambda *args: ("dihedral", args),
                    ),
                    func=SimpleNamespace(
                        CircularHarmonicFunc=lambda target, sd: ("circular", target, sd),
                    ),
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

    protocol = PyRosettaPerturbationProtocol(
        conversion_service=_DummyConversionService(),
        mutation_residue_numbers=(2,),
        score_function_name="ref2015",
        perturbation_mode="relax",
        relax_iterations=3,
        cartesian=True,
    )
    selector = object()
    task_factory = object()
    movemap = object()
    monkeypatch.setattr(
        protocol,
        "_build_mobile_selector",
        lambda pyrosetta, pose=None: selector,
    )
    monkeypatch.setattr(
        protocol,
        "_build_task_factory",
        lambda pyrosetta, packable_selector: task_factory,
    )
    monkeypatch.setattr(
        protocol,
        "_build_movemap",
        lambda pyrosetta, pose, packable_selector: movemap,
    )
    result_pose = protocol._relax_pose(_DummyPose(), _restraint_specs(), 0, "reference")

    assert create_calls == ["ref2015_cart"]
    assert relax_calls["iterations"] == 3
    assert relax_calls["task_factory"] is task_factory
    assert relax_calls["movemap"] is movemap
    assert relax_calls["disable_fixed_chi_packing"] is True
    assert relax_calls["cartesian"] is True
    assert relax_calls["min_type"] == "lbfgs_armijo_nonmonotone"
    assert relax_calls["score_function"].weights["cart_bonded"] == 0.5
    assert relax_calls["score_function"].weights["pro_close"] == 0.0
    assert isinstance(result_pose, _DummyPose)
    assert result_pose.phi(2) == -60.0
    assert result_pose.psi(2) == 140.0
    assert len(result_pose.constraints) == 2


def test_repack_pose_uses_packrotamers_trials(monkeypatch) -> None:
    create_calls: list[str] = []
    repack_calls: dict[str, object] = {}

    def create_score_function(name: str):
        create_calls.append(name)
        return _DummyScoreFunction(name)

    class _FakePackRotamersMover:
        def __init__(self, score_function, packer_task) -> None:
            repack_calls["score_function"] = score_function
            repack_calls["packer_task"] = packer_task

        def nloop(self, value: int) -> None:
            repack_calls["nloop"] = value

        def apply(self, pose) -> None:
            repack_calls["pose"] = pose

    fake_pyrosetta = SimpleNamespace(
        create_score_function=create_score_function,
        rosetta=SimpleNamespace(
            core=SimpleNamespace(
                id=SimpleNamespace(AtomID=lambda atom_index, res: (atom_index, res)),
                scoring=SimpleNamespace(
                    ScoreType=SimpleNamespace(dihedral_constraint="dihedral_constraint"),
                    constraints=SimpleNamespace(
                        DihedralConstraint=lambda *args: ("dihedral", args),
                    ),
                    func=SimpleNamespace(
                        CircularHarmonicFunc=lambda target, sd: ("circular", target, sd),
                    ),
                ),
            ),
            protocols=SimpleNamespace(
                minimization_packing=SimpleNamespace(
                    PackRotamersMover=_FakePackRotamersMover,
                )
            )
        ),
    )

    monkeypatch.setattr(
        "felp.perturbation.protocols._import_pyrosetta",
        lambda: (fake_pyrosetta, object(), object()),
    )

    protocol = PyRosettaPerturbationProtocol(
        conversion_service=_DummyConversionService(),
        mutation_residue_numbers=(2,),
        perturbation_mode="repack",
        repack_trials=4,
    )
    selector = object()
    task_factory = SimpleNamespace(
        create_task_and_apply_taskoperations=lambda pose: "packer-task"
    )
    monkeypatch.setattr(
        protocol,
        "_build_mobile_selector",
        lambda pyrosetta, pose=None: selector,
    )
    monkeypatch.setattr(
        protocol,
        "_build_task_factory",
        lambda pyrosetta, packable_selector: task_factory,
    )

    result_pose = protocol._repack_pose(_DummyPose(), _restraint_specs(), 0, "reference")

    assert create_calls == ["ref2015"]
    assert repack_calls["packer_task"] == "packer-task"
    assert repack_calls["nloop"] == 4
    assert repack_calls["pose"] is result_pose
    assert result_pose.phi(2) == -60.0
    assert result_pose.psi(2) == 140.0
    assert len(result_pose.constraints) == 2


def test_mutate_pose_skips_identity_mutation(monkeypatch) -> None:
    mutate_calls: list[tuple[int, str]] = []

    class _FakeMutateResidue:
        def __init__(self, residue_number: int, residue_name: str) -> None:
            self.residue_number = residue_number
            self.residue_name = residue_name

        def apply(self, pose) -> None:
            mutate_calls.append((self.residue_number, self.residue_name))

    monkeypatch.setattr(
        "felp.perturbation.protocols._import_pyrosetta",
        lambda: (SimpleNamespace(), object(), _FakeMutateResidue),
    )

    protocol = PyRosettaPerturbationProtocol(
        conversion_service=_DummyConversionService(),
        mutation_residue_numbers=(2,),
    )
    result_pose = protocol.mutate_pose(
        _DummyPose(residue_name="GLY"),
        (MutationSpec(residue_number=2, target_residue_name="GLY"),),
        _restraint_specs(),
    )

    assert mutate_calls == []
    assert result_pose.phi(2) == -60.0
    assert result_pose.psi(2) == 140.0


def test_mutate_pose_preserves_terminal_variants(monkeypatch) -> None:
    applied_variants: list[tuple[int, str]] = []

    class _VariantType:
        LOWER_TERMINUS_VARIANT = "LOWER_TERMINUS_VARIANT"

    class _FakeResidueType:
        def __init__(self, variants: tuple[str, ...]) -> None:
            self._variants = tuple(variants)

        def variant_types(self) -> tuple[str, ...]:
            return self._variants

    class _FakeResidue:
        def __init__(self, pose) -> None:
            self._pose = pose

        def name3(self) -> str:
            return self._pose.residue_name

        def type(self) -> _FakeResidueType:
            return _FakeResidueType(tuple(self._pose.variants))

    class _VariantPose:
        def __init__(
            self,
            residue_name: str = "GLY",
            variants: tuple[str, ...] = ("LOWER_TERMINUS_VARIANT",),
        ) -> None:
            self.residue_name = residue_name
            self.variants = list(variants)

        def clone(self):
            return _VariantPose(self.residue_name, tuple(self.variants))

        def residue(self, residue_number: int) -> _FakeResidue:
            assert residue_number == 1
            return _FakeResidue(self)

    class _FakeMutateResidue:
        def __init__(self, residue_number: int, residue_name: str) -> None:
            self.residue_number = residue_number
            self.residue_name = residue_name

        def apply(self, pose) -> None:
            assert self.residue_number == 1
            pose.residue_name = self.residue_name
            pose.variants = []

    def _add_variant(pose, variant, residue_number: int) -> None:
        assert residue_number == 1
        pose.variants.append(str(variant))
        applied_variants.append((residue_number, str(variant)))

    fake_pyrosetta = SimpleNamespace(
        rosetta=SimpleNamespace(
            core=SimpleNamespace(
                chemical=SimpleNamespace(VariantType=_VariantType),
                pose=SimpleNamespace(add_variant_type_to_pose_residue=_add_variant),
            )
        )
    )
    monkeypatch.setattr(
        "felp.perturbation.protocols._import_pyrosetta",
        lambda: (fake_pyrosetta, object(), _FakeMutateResidue),
    )

    protocol = PyRosettaPerturbationProtocol(
        conversion_service=_DummyConversionService(),
        mutation_residue_numbers=(1,),
    )
    result_pose = protocol.mutate_pose(
        _VariantPose(),
        (MutationSpec(residue_number=1, target_residue_name="TYR"),),
    )

    assert result_pose.residue_name == "TYR"
    assert result_pose.variants == ["LOWER_TERMINUS_VARIANT"]
    assert applied_variants == [(1, "LOWER_TERMINUS_VARIANT")]


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
        mutation_residue_numbers=(2,),
    )
    result_pose = protocol.enforce_residue_identity(
        _DummyPose(residue_name="HIS"),
        (MutationSpec(residue_number=2, target_residue_name="ALA", original_residue_name="HIE"),),
    )

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
        mutation_residue_numbers=(2,),
    )
    result_pose = protocol.enforce_residue_identity(
        _DummyPose(residue_name="ASP"),
        (
            MutationSpec(
                residue_number=2,
                target_residue_name="ALA",
                original_residue_name="ASH",
            ),
        ),
    )

    assert mutate_calls == [(2, "ASP_P2")]
    assert result_pose.residue(2).name3() == "ASP_P2"


def test_add_distance_constraints_uses_rosetta_sd_not_distance_spec_force_constant() -> None:
    harmonic_args: list[tuple[float, float]] = []

    class _CaptureHarmonic:
        def __init__(self, mean: float, sd: float) -> None:
            harmonic_args.append((float(mean), float(sd)))

    class _FakeResidue:
        def atom_index(self, atom_name: str) -> int:
            return 1

    class _DistPose:
        def __init__(self) -> None:
            self.added = 0

        def residue(self, residue_number: int) -> _FakeResidue:
            assert residue_number in (5, 9)
            return _FakeResidue()

        def add_constraint(self, _constraint) -> None:
            self.added += 1

    fake_pyrosetta = SimpleNamespace(
        rosetta=SimpleNamespace(
            core=SimpleNamespace(
                scoring=SimpleNamespace(
                    constraints=SimpleNamespace(
                        AtomPairConstraint=lambda *args, **kwargs: object(),
                    ),
                    func=SimpleNamespace(HarmonicFunc=_CaptureHarmonic),
                ),
                id=SimpleNamespace(AtomID=lambda atom_index, res: (atom_index, res)),
            )
        )
    )
    spec = DistanceRestraintSpec(
        label="dist",
        atom1=AtomReference(residue_number=5, atom_name="CA"),
        atom2=AtomReference(residue_number=9, atom_name="CA"),
        target_distance_angstrom=12.3,
        force_constant=50000.0,
    )
    protocol = PyRosettaPerturbationProtocol(
        conversion_service=_DummyConversionService(),
        mutation_residue_numbers=(2,),
        rosetta_distance_constraint_sd_angstrom=0.002,
    )
    pose = _DistPose()
    protocol._add_distance_constraints(fake_pyrosetta, pose, (spec,))

    assert harmonic_args == [(12.3, 0.002)]
    assert pose.added == 1


def test_enforce_residue_identity_rejects_incompatible_family(monkeypatch) -> None:
    protocol = PyRosettaPerturbationProtocol(
        conversion_service=_DummyConversionService(),
        mutation_residue_numbers=(2,),
    )

    with pytest.raises(ConversionError, match="incompatible input residue"):
        protocol.enforce_residue_identity(
            _DummyPose(residue_name="GLU"),
            (
                MutationSpec(
                    residue_number=2,
                    target_residue_name="ALA",
                    original_residue_name="HIE",
                ),
            ),
        )


def test_sequence_window_mobility_clamps_to_pose_length() -> None:
    captured_indices: list[str] = []

    class _FakeResidueIndexSelector:
        def __init__(self, residue_indices: str) -> None:
            captured_indices.append(residue_indices)

    fake_pyrosetta = SimpleNamespace(
        rosetta=SimpleNamespace(
            core=SimpleNamespace(
                select=SimpleNamespace(
                    residue_selector=SimpleNamespace(
                        ResidueIndexSelector=_FakeResidueIndexSelector,
                    )
                )
            )
        )
    )
    protocol = PyRosettaPerturbationProtocol(
        conversion_service=_DummyConversionService(),
        mutation_residue_numbers=(4, 10),
        mobility=MobilityConfig(mode="sequence_window", sequence_window=1),
    )

    pose = SimpleNamespace(total_residue=lambda: 10)
    protocol._build_mobile_selector(fake_pyrosetta, pose)

    assert captured_indices == ["3,4,5,9,10"]


class _BareMultiResiduePose:
    def __init__(self, names_by_residue: dict[int, str]) -> None:
        self._names = dict(names_by_residue)

    def clone(self) -> "_BareMultiResiduePose":
        return _BareMultiResiduePose(dict(self._names))

    def residue(self, residue_number: int):
        name = self._names[residue_number]
        return SimpleNamespace(name3=lambda: name.upper(), atom_index=lambda _name: 3)


def test_mutate_pose_orders_mutations_by_configuration(monkeypatch) -> None:
    applied_order: list[int] = []

    class _FakeMutateResidue:
        def __init__(self, residue_number: int, residue_name: str) -> None:
            self.residue_number = residue_number
            self.residue_name = residue_name

        def apply(self, pose) -> None:
            applied_order.append(self.residue_number)

    monkeypatch.setattr(
        "felp.perturbation.protocols._import_pyrosetta",
        lambda: (SimpleNamespace(), object(), _FakeMutateResidue),
    )

    protocol = PyRosettaPerturbationProtocol(
        conversion_service=_DummyConversionService(),
        mutation_residue_numbers=(7, 2),
    )
    mutations = (
        MutationSpec(residue_number=7, target_residue_name="ALA"),
        MutationSpec(residue_number=2, target_residue_name="GLY"),
    )
    pose = _BareMultiResiduePose({7: "GLY", 2: "ALA"})
    protocol.mutate_pose(pose, mutations, ())

    assert applied_order == [7, 2]


def test_radius_mobility_uses_minimum_atom_distance() -> None:
    # Residue 3 contacts residue 1 even though its first atom is far away.
    points = {1: [0.0], 2: [30.0], 3: [20.0, 2.0], 4: [40.0]}
    class Residue:
        def __init__(self, coordinates):
            self.coordinates = coordinates
        def natoms(self):
            return len(self.coordinates)
        def xyz(self, index):
            return SimpleNamespace(x=self.coordinates[index - 1], y=0.0, z=0.0)
    pose = SimpleNamespace(total_residue=lambda: 4, residue=lambda n: Residue(points[n]))
    selectors = SimpleNamespace(ResidueIndexSelector=lambda text: text)
    pyrosetta = SimpleNamespace(rosetta=SimpleNamespace(core=SimpleNamespace(
        select=SimpleNamespace(residue_selector=selectors))))
    protocol = PyRosettaPerturbationProtocol(
        conversion_service=_DummyConversionService(), mutation_residue_numbers=(1, 4),
        mobility=MobilityConfig(mode="radius", radius_angstrom=3),
    )
    assert protocol._build_mobile_selector(pyrosetta, pose) == "1,3,4"

def test_custom_dihedral_projection_respects_locked_native_torsions(monkeypatch):
    protocol = PyRosettaPerturbationProtocol(
        conversion_service=_DummyConversionService(), mutation_residue_numbers=(162,),
    )
    monkeypatch.setattr(protocol, "_measure_dihedral_from_pose", lambda pose, spec: 10.0)
    pose = SimpleNamespace(total_residue=lambda: 200)
    custom = DihedralRestraintSpec(
        label="aloop", kind="dihedral", residue_numbers=(159, 160, 161, 162),
        atom_names=("CA", "CA", "CA", "CA"),
        target_angle_degrees=20.0, force_constant=10000.0,
    )
    locked = tuple(
        DihedralRestraintSpec(
            label=kind, kind=kind, residue_numbers=(161, 161, 161, 161),
            atom_names=("N", "CA", "C", "N"),
            target_angle_degrees=0.0, force_constant=10000.0,
        ) for kind in ("phi", "psi")
    )
    with pytest.raises(ConversionError, match="No unlocked native backbone torsion"):
        protocol._enforce_custom_dihedral_on_pose(pose, custom, 20.0, (*locked, custom))


def test_enforce_custom_dihedral_projects_with_native_phi(monkeypatch) -> None:
    """A pseudodihedral is restored through a real nearby backbone torsion."""

    class _PseudoDihedralPose:
        def __init__(self, phi: float = 80.148, psi: float = -45.0) -> None:
            self._phi = phi
            self._psi = psi

        def clone(self):
            return _PseudoDihedralPose(self._phi, self._psi)

        def total_residue(self) -> int:
            return 200

        def phi(self, residue_number: int) -> float:
            assert residue_number == 161
            return self._phi

        def set_phi(self, residue_number: int, value: float) -> None:
            assert residue_number == 161
            self._phi = value

        def psi(self, residue_number: int) -> float:
            assert residue_number == 161
            return self._psi

        def set_psi(self, residue_number: int, value: float) -> None:
            assert residue_number == 161
            self._psi = value

        def set_xyz(self, *args, **kwargs) -> None:
            raise AssertionError("Cartesian coordinates must not be set directly")

    pose = _PseudoDihedralPose()
    protocol = PyRosettaPerturbationProtocol(
        conversion_service=_DummyConversionService(),
        mutation_residue_numbers=(2,),
    )
    monkeypatch.setattr(
        protocol,
        "_measure_dihedral_from_pose",
        lambda observed_pose, spec: observed_pose._phi,
    )
    protocol._enforce_target_backbone_cvs(
        pose,
        (
            DihedralRestraintSpec(
                label="aloop",
                kind="dihedral",
                residue_numbers=(159, 160, 161, 162),
                atom_names=("CA", "CA", "CA", "CA"),
                target_angle_degrees=80.095,
                force_constant=10000.0,
            ),
        ),
    )
    assert pose._phi == pytest.approx(80.095, abs=1.0e-6)
    assert pose._psi == -45.0


def test_enforce_custom_dihedral_rejects_unprojectable_pseudodihedral(
    monkeypatch,
) -> None:
    """A nonresponsive native DOF must fail rather than accept CV drift."""

    class _NonresponsivePose:
        def __init__(self, phi: float = 10.0, psi: float = 20.0) -> None:
            self._phi = phi
            self._psi = psi

        def clone(self):
            return _NonresponsivePose(self._phi, self._psi)

        def total_residue(self) -> int:
            return 200

        def phi(self, residue_number: int) -> float:
            return self._phi

        def set_phi(self, residue_number: int, value: float) -> None:
            self._phi = value

        def psi(self, residue_number: int) -> float:
            return self._psi

        def set_psi(self, residue_number: int, value: float) -> None:
            self._psi = value

    protocol = PyRosettaPerturbationProtocol(
        conversion_service=_DummyConversionService(),
        mutation_residue_numbers=(2,),
    )
    monkeypatch.setattr(
        protocol,
        "_measure_dihedral_from_pose",
        lambda observed_pose, spec: 80.148,
    )
    restraint = DihedralRestraintSpec(
        label="aloop",
        kind="dihedral",
        residue_numbers=(159, 160, 161, 162),
        atom_names=("CA", "CA", "CA", "CA"),
        target_angle_degrees=80.095,
        force_constant=10000.0,
    )

    with pytest.raises(ConversionError, match="Could not project custom dihedral"):
        protocol._enforce_target_backbone_cvs(_NonresponsivePose(), (restraint,))


def _distance_cv_pose(separation_angstrom: float):
    class _Xyz:
        def __init__(self, x: float) -> None:
            self.x = x
            self.y = 0.0
            self.z = 0.0

    class _Residue:
        def __init__(self, x: float) -> None:
            self._x = x

        def xyz(self, atom_name: str) -> _Xyz:
            return _Xyz(self._x)

    class _Pose:
        def residue(self, residue_number: int) -> _Residue:
            return _Residue(0.0 if residue_number == 5 else separation_angstrom)

    return _Pose()


def _distance_restraint(target: float) -> DistanceRestraintSpec:
    return DistanceRestraintSpec(
        label="ac_diff__negative",
        atom1=AtomReference(residue_number=5, atom_name="NZ"),
        atom2=AtomReference(residue_number=9, atom_name="OE1"),
        target_distance_angstrom=target,
        force_constant=10000.0,
    )


def test_rosetta_stage_rejects_distance_cv_drift() -> None:
    """Distance-CV drift must fail at the Rosetta stage, not silently reach OpenMM."""

    protocol = PyRosettaPerturbationProtocol(
        conversion_service=_DummyConversionService(),
        mutation_residue_numbers=(2,),
        distance_constraint_tolerance_angstrom=0.05,
    )

    with pytest.raises(ConversionError, match="ac_diff__negative"):
        protocol._assert_locked_backbone_cvs(
            _distance_cv_pose(4.9),
            (_distance_restraint(4.75),),
            context="Rosetta FastRelax",
        )


def test_rosetta_stage_accepts_distance_cv_within_tolerance() -> None:
    protocol = PyRosettaPerturbationProtocol(
        conversion_service=_DummyConversionService(),
        mutation_residue_numbers=(2,),
        distance_constraint_tolerance_angstrom=0.05,
    )

    protocol._assert_locked_backbone_cvs(
        _distance_cv_pose(4.77),
        (_distance_restraint(4.75),),
        context="Rosetta FastRelax",
    )
