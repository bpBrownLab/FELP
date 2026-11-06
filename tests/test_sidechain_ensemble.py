"""Tests for optional sidechain-ensemble sampling, aggregation, and I/O."""

from __future__ import annotations

import argparse
import math
from types import SimpleNamespace

import pytest

from felp.cli.parser import build_sidechain_ensemble_config
from felp.core import (
    ConfigurationError,
    EnergyResult,
    FrameResult,
    ReweightingError,
    SidechainEnsembleConfig,
    SidechainEnsembleMember,
    frame_reweighting_delta,
)
from felp.energy.evaluators import OpenMMEnergyEvaluator
from felp.io.results import (
    ResultsWriter,
    attach_sidechain_ensemble_members,
    load_sidechain_ensemble_members,
)
from felp.core import ReweightingError
from felp.sidechain_ensemble import (
    SampledPose,
    attach_ensemble_to_frame,
    classify_rotamer_assignment,
    configure_sidechain_mover,
    members_from_energy_results,
    proposal_quadrature_log_z,
    require_proposal_density,
    run_jump_rotamer_chain,
)
from felp.sidechain_ensemble.aggregation import local_free_energy_kj_mol


class _FakeResidue:
    def __init__(self, chi: float) -> None:
        self._chi = chi

    def nchi(self) -> int:
        return 1

    def chi(self, _index: int) -> float:
        return self._chi


class _FakePose:
    def __init__(self, chi: float = 60.0) -> None:
        self.chi_value = chi
        self.moves = 0

    def clone(self) -> "_FakePose":
        cloned = _FakePose(self.chi_value)
        cloned.moves = self.moves
        return cloned

    def residue(self, _number: int) -> _FakeResidue:
        return _FakeResidue(self.chi_value)


def _frame(
    index: int,
    *,
    delta_u: float = 0.0,
    trajectory_id: int = 0,
    phi: float = -60.0,
    psi: float = 0.0,
    reweighting_delta: float | None = None,
) -> FrameResult:
    return FrameResult(
        frame_index=index,
        stride_number=index,
        microstate_id="s0",
        microstate_label="s0",
        phi_degrees=phi,
        psi_degrees=psi,
        reference_energy_kj_mol=10.0,
        perturbed_energy_kj_mol=10.0 + delta_u,
        delta_u_kj_mol=delta_u,
        reference_relaxed_phi_degrees=phi,
        reference_relaxed_psi_degrees=psi,
        perturbed_relaxed_phi_degrees=phi,
        perturbed_relaxed_psi_degrees=psi,
        trajectory_id=trajectory_id,
        source_frame_index=index,
        reweighting_delta_kj_mol=reweighting_delta,
    )


def test_default_sidechain_ensemble_is_disabled() -> None:
    config = SidechainEnsembleConfig()
    assert config.mode == "off"
    assert not config.enabled
    assert not config.replaces_production_weight


def test_invalid_sampler_mode_combinations() -> None:
    with pytest.raises(ConfigurationError):
        SidechainEnsembleConfig(mode="proposal-quadrature", sampler="mh")
    with pytest.raises(ConfigurationError):
        SidechainEnsembleConfig(mode="diagnostic-mh", sampler="jump-rotamer")
    with pytest.raises(ConfigurationError):
        SidechainEnsembleConfig(mode="empirical-mh", sampler="jump-rotamer")


def test_from_dict_auto_selects_jump_rotamer_for_quadrature() -> None:
    config = SidechainEnsembleConfig.from_dict({"mode": "proposal-quadrature"})
    assert config.sampler == "jump-rotamer"


def test_cli_auto_sampler_and_explicit_invalid_combo() -> None:
    auto = build_sidechain_ensemble_config(
        argparse.Namespace(sidechain_ensemble_mode="proposal-quadrature")
    )
    assert auto.sampler == "jump-rotamer"
    with pytest.raises(ConfigurationError):
        build_sidechain_ensemble_config(
            argparse.Namespace(
                sidechain_ensemble_mode="proposal-quadrature",
                sidechain_ensemble_sampler="mh",
            )
        )






def test_configure_sidechain_mover_enables_detailed_balance() -> None:
    mover = SimpleNamespace()
    calls: dict[str, object] = {}

    def _set_balance(value: bool) -> None:
        calls["detailed_balance"] = value

    mover.set_preserve_detailed_balance = _set_balance
    mover.set_prob_uniform = lambda value: calls.setdefault("uniform", value)
    configure_sidechain_mover(
        mover,
        SidechainEnsembleConfig(mode="proposal-quadrature", prob_uniform=0.2),
    )
    assert calls["detailed_balance"] is True
    assert calls["uniform"] == 0.2


def test_proposal_quadrature_recovers_known_partition_sum() -> None:
    beta = 0.4
    energies = [0.0, 2.0, 5.0]
    densities = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    exact = math.log(sum(math.exp(-beta * energy) for energy in energies))
    estimated = proposal_quadrature_log_z(energies, densities, beta)
    assert estimated == pytest.approx(exact, abs=1.0e-12)


def test_missing_or_zero_proposal_density_fails() -> None:
    with pytest.raises(ReweightingError):
        require_proposal_density(None, index=0)
    with pytest.raises(ReweightingError):
        proposal_quadrature_log_z([1.0], [0.0], 0.4)




def test_jump_rotamer_records_independent_draws() -> None:
    config = SidechainEnsembleConfig(
        mode="proposal-quadrature",
        sampler="jump-rotamer",
        retained_samples=3,
    )
    draws = []

    def _draw(pose: _FakePose) -> float:
        pose.chi_value = 60.0 + 10.0 * len(draws)
        draws.append(pose.chi_value)
        return 0.25

    members = run_jump_rotamer_chain(
        start_pose=_FakePose(60.0),
        draw_fn=_draw,
        config=config,
        branch="perturbed",
        chain_id=1,
        residue_numbers=(1,),
        seed=3,
    )
    assert len(members) == 3
    assert [member.proposal_density for member in members] == [0.25, 0.25, 0.25]
    assert [member.chi_degrees for member in members] == [(60.0,), (70.0,), (80.0,)]
    assert classify_rotamer_assignment((180.0,)) == "t"


def test_jump_rotamer_rejects_non_positive_density() -> None:
    config = SidechainEnsembleConfig(
        mode="proposal-quadrature",
        sampler="jump-rotamer",
        retained_samples=3,
    )
    densities = iter([0.5, 0.0, float("nan")])

    members = run_jump_rotamer_chain(
        start_pose=_FakePose(60.0),
        draw_fn=lambda _pose: next(densities),
        config=config,
        branch="reference",
        chain_id=0,
        residue_numbers=(1,),
        seed=3,
    )
    assert [member.proposal_density for member in members] == [0.5, None, None]


def test_parent_row_omits_ensemble_members() -> None:
    members = (
        SidechainEnsembleMember(
            frame_index=7,
            branch="reference",
            chain_id=0,
            step=1,
            energy_kj_mol=1.0,
            proposal_density=0.5,
        ),
    )
    frame = attach_ensemble_to_frame(
        _frame(7, delta_u=1.5),
        members
        + (
            SidechainEnsembleMember(
                frame_index=7,
                branch="perturbed",
                chain_id=0,
                step=1,
                energy_kj_mol=2.0,
                proposal_density=0.5,
            ),
        ),
        config=SidechainEnsembleConfig(
            mode="proposal-quadrature",
            sampler="jump-rotamer",
        ),
        beta=0.4,
    )
    row = frame.to_row()
    assert "ensemble_members" not in row
    assert frame.reweighting_delta_kj_mol == pytest.approx(frame.local_free_energy_kj_mol)
    assert frame_reweighting_delta(frame) == pytest.approx(frame.local_free_energy_kj_mol)
    assert frame.delta_u_kj_mol == 1.5


def test_sidecar_roundtrip_and_merge(tmp_path) -> None:
    writer = ResultsWriter(
        type("Output", (), {"output_dir": str(tmp_path), "debug_frame_outputs": False})()
    )
    first = [
        SidechainEnsembleMember(
            frame_index=1,
            branch="reference",
            chain_id=0,
            step=1,
            energy_kj_mol=1.2,
            chi_degrees=(60.0, 180.0),
        )
    ]
    writer.write_sidechain_ensemble_members(first)
    writer.merge_sidechain_ensemble_members(
        [
            SidechainEnsembleMember(
                frame_index=1,
                branch="perturbed",
                chain_id=0,
                step=1,
                energy_kj_mol=2.4,
            ),
            SidechainEnsembleMember(
                frame_index=2,
                branch="reference",
                chain_id=0,
                step=1,
                energy_kj_mol=0.5,
            ),
        ]
    )
    loaded = load_sidechain_ensemble_members(writer.sidechain_ensemble_path)
    assert {member.frame_index for member in loaded} == {1, 2}
    assert sum(member.branch == "reference" and member.frame_index == 1 for member in loaded) == 0
    frames = attach_sidechain_ensemble_members([_frame(1), _frame(2)], loaded)
    assert len(frames[0].ensemble_members) == 1
    assert frames[0].ensemble_members[0].branch == "perturbed"


def test_evaluate_batch_reuses_one_session() -> None:
    class _Dummy:
        def __init__(self) -> None:
            self.sessions: list[object] = []

        def open_minimization_session(self, structure, restraint_specs, **_kwargs):
            session = SimpleNamespace(minimize_calls=0)

            def _minimize(item) -> EnergyResult:
                session.minimize_calls += 1
                return EnergyResult(
                    backend="openmm",
                    energy_kj_mol=float(item),
                    minimized_coordinates_angstrom=(),
                )

            session.minimize = _minimize
            session.close = lambda: setattr(session, "closed", True)
            self.sessions.append(session)
            return session

    dummy = _Dummy()
    results = OpenMMEnergyEvaluator.evaluate_batch(
        dummy,
        [1.0, 2.0, 3.0],
        (),
        template_structure=0.0,
    )
    assert [result.energy_kj_mol for result in results] == [1.0, 2.0, 3.0]
    assert len(dummy.sessions) == 1
    assert dummy.sessions[0].minimize_calls == 3
    assert dummy.sessions[0].closed is True


def test_members_from_energy_results_keep_duplicates() -> None:
    sampled = [
        SampledPose(
            branch="reference",
            chain_id=0,
            step=index,
            pose=None,
            seed=0,
            rotamer_assignment="p",
        )
        for index in (1, 2)
    ]
    energies = [
        EnergyResult(backend="openmm", energy_kj_mol=1.0, minimized_coordinates_angstrom=())
        for _ in sampled
    ]
    members = members_from_energy_results(sampled, energies, frame_index=9)
    assert len(members) == 2
    assert members[0].rotamer_assignment == members[1].rotamer_assignment






def test_local_free_energy_sign() -> None:
    delta = local_free_energy_kj_mol(log_z_perturbed=1.0, log_z_reference=0.0, beta=0.5)
    assert delta == pytest.approx(-2.0)
