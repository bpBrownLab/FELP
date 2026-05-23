"""Tests for overlap diagnostic thresholds."""

from __future__ import annotations

from felp.reweighting.overlap_support import (
    LOW_ESS_ABSOLUTE_THRESHOLDS,
    state_ess_threshold,
    state_has_unreliable_local_support,
)


def test_low_ess_absolute_thresholds_are_documented_nominals() -> None:
    assert LOW_ESS_ABSOLUTE_THRESHOLDS == (10, 50)


def test_state_ess_threshold_scales_with_frame_count() -> None:
    assert state_ess_threshold(100, min_state_ess=5.0, relative_fraction=0.1) == 10.0
    assert state_ess_threshold(20, min_state_ess=5.0, relative_fraction=0.1) == 5.0


def test_unreliable_local_support() -> None:
    assert state_has_unreliable_local_support(4.0, 100, min_state_ess=5.0)
    assert not state_has_unreliable_local_support(15.0, 100, min_state_ess=5.0)
    assert state_has_unreliable_local_support(6.0, 20, min_state_ess=5.0)
