"""Global frame-level QC for frame SNIS reweighting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from felp.core import FrameResult, ReweightingQCConfig
from felp.reweighting.aggregator import _effective_sample_size, psis_smooth_weights


@dataclass(frozen=True)
class FrameQCResult:
    """Outcome of global frame QC filtering."""

    frames: list[FrameResult]
    weights: np.ndarray
    excluded_frame_indices: tuple[int, ...]
    global_ess: float
    global_ess_fraction: float
    max_normalized_weight: float
    status: str


def apply_global_frame_qc(
    frames: list[FrameResult],
    weights: np.ndarray,
    *,
    qc_config: ReweightingQCConfig,
    max_excluded_frames: int,
) -> FrameQCResult:
    """Optionally exclude dominant global frames and report global ESS."""

    remaining_frames = list(frames)
    remaining_weights = np.asarray(weights, dtype=float)
    excluded: list[int] = []

    if qc_config.weight_treatment == "psis-smoothed" and remaining_weights.size:
        remaining_weights, _ = psis_smooth_weights(remaining_weights)

    while (
        qc_config.policy == "exclude-frames"
        and remaining_weights.size
        and len(excluded) < max_excluded_frames
    ):
        weight_sum = float(remaining_weights.sum())
        if weight_sum <= 0.0:
            break
        normalized = remaining_weights / weight_sum
        dominant_index = int(np.argmax(normalized))
        if float(normalized[dominant_index]) <= qc_config.max_frame_weight_fraction:
            break
        dominant_frame = remaining_frames[dominant_index]
        excluded.append(dominant_frame.frame_index)
        remaining_frames = [
            frame
            for index, frame in enumerate(remaining_frames)
            if index != dominant_index
        ]
        remaining_weights = np.delete(remaining_weights, dominant_index)

    weight_sum = float(remaining_weights.sum())
    normalized = (
        remaining_weights / weight_sum if weight_sum > 0.0 else remaining_weights
    )
    global_ess = _effective_sample_size(remaining_weights)
    frame_count = len(remaining_frames)
    return FrameQCResult(
        frames=remaining_frames,
        weights=remaining_weights,
        excluded_frame_indices=tuple(excluded),
        global_ess=global_ess,
        global_ess_fraction=global_ess / frame_count if frame_count else 0.0,
        max_normalized_weight=float(normalized.max()) if normalized.size else 0.0,
        status="filtered" if excluded else "ok",
    )
