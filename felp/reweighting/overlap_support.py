"""Overlap diagnostic thresholds and state-local support helpers."""

from __future__ import annotations

# Nominal absolute Kish-ESS cutoffs (independent-sample scale) for reporting how much
# target probability sits in weakly supported states. ESS ≈ 10 is a common IS rule of
# thumb for minimally useful samples; 50 is more lenient. See docs/theory.md.
LOW_ESS_ABSOLUTE_THRESHOLDS: tuple[int, ...] = (10, 50)

# Relative floor: state i is unreliable if ESS_i < max(min_state_ess, rho * n_i).
# rho = 0.1 means "fewer than ~10% of frames contribute as independent samples."
LOW_ESS_RELATIVE_FRACTION: float = 0.1

# Backward-compatible alias used across the reweighting stack.
_LOW_ESS_THRESHOLDS = LOW_ESS_ABSOLUTE_THRESHOLDS


def state_ess_threshold(
    frame_count: int,
    *,
    min_state_ess: float = 5.0,
    relative_fraction: float = LOW_ESS_RELATIVE_FRACTION,
) -> float:
    """Minimum ESS for adequate within-state support at this frame count."""

    if frame_count <= 0:
        return float("inf")
    return max(float(min_state_ess), relative_fraction * float(frame_count))


def state_has_unreliable_local_support(
    effective_sample_size: float,
    frame_count: int,
    *,
    min_state_ess: float = 5.0,
    relative_fraction: float = LOW_ESS_RELATIVE_FRACTION,
) -> bool:
    """Return whether state-local weights are too degenerate for trust."""

    return effective_sample_size < state_ess_threshold(
        frame_count,
        min_state_ess=min_state_ess,
        relative_fraction=relative_fraction,
    )
