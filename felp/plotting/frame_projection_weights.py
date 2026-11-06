"""Frame-level weights for CV projection plots."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from felp.core import frame_reweighting_delta
from felp.reweighting.psis import treat_weights

from felp.reweighting.frame_weights import load_frame_weights_csv

__all__ = [
    "frame_projection_weight_detail",
    "load_frame_weights_csv",
    "per_frame_projection_weight",
    "state_reweight_factors_from_summary_rows",
    "treated_weights_for_frames",
]


def state_reweight_factors_from_summary_rows(
    rows: list[dict[str, str]],
) -> dict[str, float]:
    """Read per-state reweight factors recorded in ``microstate_summary.csv``."""

    factors: dict[str, float] = {}
    for row in rows:
        state_id = row.get("state_id") or row.get("label")
        if not state_id:
            continue
        raw = row.get("reweight_factor")
        if raw in (None, ""):
            continue
        factors[str(state_id)] = float(raw)
    return factors


def per_frame_projection_weight(
    *,
    reference_probability: float,
    treated_weight: float,
    local_mean_treated: float,
    state_reweight_factor: float | None,
    frame_weight_override: float | None,
) -> float:
    """Return the target-channel frame weight used in CV histogram projection."""

    if frame_weight_override is not None:
        return float(frame_weight_override)
    if local_mean_treated <= 0.0 or reference_probability <= 0.0:
        return 0.0
    scale = 1.0
    if state_reweight_factor is not None:
        scale = float(state_reweight_factor) / local_mean_treated
    return float(reference_probability * treated_weight * scale)


def treated_weights_for_frames(
    frames: list[Any],
    *,
    beta: float,
    global_delta_shift: float,
    weight_treatment: str,
) -> np.ndarray:
    """Match :mod:`felp.reweighting.prep` local Boltzmann weights for a frame list."""

    if not frames:
        return np.asarray([], dtype=float)
    delta_values = np.asarray([frame_reweighting_delta(frame) for frame in frames], dtype=float)
    local_delta_shift = float(delta_values.min())
    treated = np.exp(-beta * (delta_values - local_delta_shift))
    if weight_treatment != "raw":
        treated, _ = treat_weights(treated, weight_treatment, log_weights=-beta * (delta_values - local_delta_shift))
    return np.exp(-beta * (local_delta_shift - global_delta_shift)) * treated


def frame_projection_weight_detail(
    *,
    reweighting_method: str,
    used_frame_weights_csv: bool,
) -> str:
    """Short description for ``plot_summary.json``."""

    if used_frame_weights_csv:
        return (
            "Frame weights loaded from frame_weights.csv (frame-level SNIS export)."
        )
    method = reweighting_method or "independent"
    if method == "independent":
        return (
            "w_ij proportional to pi_reference(i) * "
            "exp(-beta * delta_u_ij) / n_i with state mass pi_reference(i) * f_i."
        )
    return (
        f"w_ij proportional to pi_reference(i) * exp(-beta * delta_u_ij) / n_i scaled "
        f"by state reweight_factor f_i / <exp(-beta * delta_u)>_i to match "
        f"reweighting_method={method!r} microstate_summary masses."
    )


def resolve_frame_weights_csv_path(
    *,
    microstate_summary_csv: str,
    explicit_path: str | None,
) -> str | None:
    """Resolve an optional ``frame_weights.csv`` beside the microstate summary."""

    if explicit_path:
        return explicit_path if os.path.isfile(explicit_path) else None
    summary_dir = os.path.dirname(os.path.abspath(microstate_summary_csv))
    candidate = os.path.join(summary_dir, "frame_weights.csv")
    return candidate if os.path.isfile(candidate) else None
