"""Tests for method-aware frame CV projection weights."""

from __future__ import annotations

import numpy as np
import pytest

from felp.core import FrameResult
from felp.plotting.felp import _collect_cv_projection_samples
from felp.plotting.frame_projection_weights import per_frame_projection_weight
from felp.reweighting.frame_weights import (
    build_state_factor_frame_weight_records,
    normalize_frame_weight_records,
)


def _frame(
    frame_index: int,
    state_id: str,
    *,
    phi: float = 0.0,
    psi: float = 0.0,
    delta_u: float = 0.0,
) -> FrameResult:
    return FrameResult(
        frame_index=frame_index,
        stride_number=frame_index,
        microstate_id=state_id,
        microstate_label=state_id,
        phi_degrees=phi,
        psi_degrees=psi,
        reference_energy_kj_mol=0.0,
        perturbed_energy_kj_mol=delta_u,
        delta_u_kj_mol=delta_u,
        reference_relaxed_phi_degrees=phi,
        reference_relaxed_psi_degrees=psi,
        perturbed_relaxed_phi_degrees=phi,
        perturbed_relaxed_psi_degrees=psi,
    )


def test_per_frame_projection_weight_scales_by_state_reweight_factor() -> None:
    weight = per_frame_projection_weight(
        reference_probability=0.2,
        treated_weight=2.0,
        local_mean_treated=1.0,
        state_reweight_factor=1.5,
        frame_weight_override=None,
    )
    assert weight == pytest.approx(0.6)


def test_build_state_factor_frame_weight_records_matches_state_mass() -> None:
    frames = {
        "a": [_frame(0, "a", delta_u=0.0), _frame(1, "a", delta_u=1.0)],
        "b": [_frame(2, "b", delta_u=0.5)],
    }
    treated = {
        "a": np.asarray([1.0, 3.0], dtype=float),
        "b": np.asarray([2.0], dtype=float),
    }
    reference = {"a": 0.6, "b": 0.4}
    factors = {"a": 1.2, "b": 0.8}
    records = normalize_frame_weight_records(
        build_state_factor_frame_weight_records(
            frames_by_state=frames,
            treated_weights_by_state=treated,
            reference_probability_by_state=reference,
            state_reweight_factors=factors,
        )
    )
    mass_by_state: dict[str, float] = {}
    for record in records:
        state_id = str(record["microstate_id"])
        mass_by_state[state_id] = mass_by_state.get(state_id, 0.0) + float(record["weight"])

    normalization = sum(reference[state_id] * factors[state_id] for state_id in reference)
    expected = {
        state_id: reference[state_id] * factors[state_id] / normalization
        for state_id in reference
    }
    for state_id, target_mass in expected.items():
        assert mass_by_state[state_id] == pytest.approx(target_mass)


def test_collect_cv_projection_samples_preserves_neighbor_pool_state_mass() -> None:
    frames = [
        _frame(0, "s0", phi=-60.0, psi=120.0, delta_u=0.0),
        _frame(1, "s0", phi=-55.0, psi=125.0, delta_u=2.0),
        _frame(2, "s1", phi=60.0, psi=-30.0, delta_u=1.0),
    ]
    reference = {"s0": 0.7, "s1": 0.3}
    factors = {"s0": 1.1, "s1": 0.9}
    sampled = _collect_cv_projection_samples(
        frame_results=frames,
        reference_probability_by_state=reference,
        temperature_kelvin=300.0,
        weight_treatment="raw",
        max_points=10_000,
        x_name="phi",
        y_name="psi",
        state_reweight_factors=factors,
        reweighting_method="neighbor_pool",
    )
    assert sampled is not None
    _x_values, _y_values, _reference_weights, reweighted_weights = sampled
    mass_by_state = {"s0": 0.0, "s1": 0.0}
    cursor = 0
    for state_id in ("s0", "s1"):
        state_frames = [frame for frame in frames if frame.microstate_id == state_id]
        state_weights = reweighted_weights[cursor : cursor + len(state_frames)]
        cursor += len(state_frames)
        mass_by_state[state_id] = float(np.sum(state_weights))
    normalization = sum(reference[state_id] * factors[state_id] for state_id in reference)
    for state_id in reference:
        expected = reference[state_id] * factors[state_id] / normalization
        assert mass_by_state[state_id] == pytest.approx(expected, rel=1.0e-6)
