"""Frame-level reweighting weights for exports and CV projection."""

from __future__ import annotations

import csv
from typing import Any

import numpy as np

from felp.core import FrameResult


def load_frame_weights_csv(path: str) -> dict[int, float]:
    """Load ``frame_index -> weight`` from a frame-level reweighting export."""

    weights: dict[int, float] = {}
    with open(path, encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            frame_index = int(row["frame_index"])
            weights[frame_index] = float(row["weight"])
    return weights


def build_state_factor_frame_weight_records(
    *,
    frames_by_state: dict[str, list[FrameResult]],
    treated_weights_by_state: dict[str, np.ndarray],
    reference_probability_by_state: dict[str, float],
    state_reweight_factors: dict[str, float],
) -> list[dict[str, object]]:
    """Build per-frame weights from local Boltzmann factors and state reweight factors.

    Within each state, weights follow the same local Boltzmann factors used in
    preparation. Across states, masses match ``pi_reference(i) * f_i`` where ``f_i``
    is the recorded state reweight factor.
    """

    records: list[dict[str, object]] = []
    for state_id, frames in frames_by_state.items():
        treated = treated_weights_by_state.get(state_id)
        if treated is None or treated.size == 0:
            continue
        local_mean = float(np.mean(treated))
        state_factor = float(state_reweight_factors.get(state_id, 0.0))
        reference_probability = float(reference_probability_by_state.get(state_id, 0.0))
        if local_mean <= 0.0 or reference_probability <= 0.0 or state_factor <= 0.0:
            continue
        scale = state_factor / local_mean
        frame_count = len(frames)
        for frame, local_weight in zip(frames, treated, strict=True):
            records.append(
                {
                    "frame_index": frame.frame_index,
                    "microstate_id": state_id,
                    "delta_u_kj_mol": frame.delta_u_kj_mol,
                    "phi_degrees": frame.phi_degrees,
                    "psi_degrees": frame.psi_degrees,
                    "weight": float(
                        reference_probability * local_weight * scale / frame_count
                    ),
                }
            )
    return records


def normalize_frame_weight_records(
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Globally normalize frame weights to sum to one."""

    total = sum(float(record["weight"]) for record in records)
    if total <= 0.0:
        return records
    return [
        {**record, "weight": float(record["weight"]) / total} for record in records
    ]


def frame_weights_from_prep(
    prep: Any,
    reweight_factors: dict[str, float],
) -> list[dict[str, object]]:
    """Export normalized frame weights for a state-factor reweighting outcome."""

    records = build_state_factor_frame_weight_records(
        frames_by_state={
            state_id: state.frames for state_id, state in prep.state_data.items()
        },
        treated_weights_by_state={
            state_id: state.treated_weights
            for state_id, state in prep.state_data.items()
        },
        reference_probability_by_state={
            state_id: state.reference_probability
            for state_id, state in prep.state_data.items()
        },
        state_reweight_factors=reweight_factors,
    )
    return normalize_frame_weight_records(records)
