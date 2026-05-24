"""Tests for KL heatmap display threshold masking."""

from __future__ import annotations

import numpy as np
import pytest

from felp.core import ConfigurationError
from felp.plotting.common import (
    apply_kl_display_threshold,
    prepare_kl_display_grid,
    resolve_kl_colorbar_limits,
    symmetric_kl_imbalance_mathtext_label,
)


def test_apply_kl_display_threshold_zeros_small_cells() -> None:
    grid = np.asarray([[2.0e-6, -3.0e-6], [1.0e-4, 0.0]])
    masked, metadata = apply_kl_display_threshold(grid, threshold=1.0e-5)
    assert masked[0, 0] == 0.0
    assert masked[0, 1] == 0.0
    assert masked[1, 0] == pytest.approx(1.0e-4)
    assert metadata["kl_display_cells_zeroed"] == 3


def test_resolve_kl_colorbar_limits_symmetric_from_vmax() -> None:
    assert resolve_kl_colorbar_limits(None, 1.0e-4) == (-1.0e-4, 1.0e-4)


def test_resolve_kl_colorbar_limits_explicit_range() -> None:
    assert resolve_kl_colorbar_limits(-2.0, 3.0) == (-2.0, 3.0)


def test_resolve_kl_colorbar_limits_rejects_invalid_range() -> None:
    with pytest.raises(ConfigurationError):
        resolve_kl_colorbar_limits(1.0, -1.0)


def test_symmetric_kl_imbalance_mathtext_label_uses_matplotlib_math() -> None:
    label = symmetric_kl_imbalance_mathtext_label("p_MD", "p_rw", unit_label="kcal/mol")
    assert label.startswith("$")
    assert label.endswith("(kcal/mol)")
    assert r"\frac{1}{2}" in label
    assert r"\ln\frac" in label
    assert r"p_{\mathrm{MD}}" in label
    assert r"p_{\mathrm{rw}}" in label


def test_prepare_kl_display_grid_masks_after_kbt_scaling() -> None:
    grid = np.asarray([[1.0e-9, 0.0]])
    display, label, metadata = prepare_kl_display_grid(
        grid,
        temperature_kelvin=310.0,
        scale_by_kbt=True,
        kbt_energy_scale=1.0 / 4.184,
        kbt_energy_unit_label="kcal/mol",
        display_threshold=1.0e-5,
        probability_labels=("p_MD", "p_rw"),
    )
    assert display[0, 0] == 0.0
    assert label.startswith("$")
    assert r"\frac{1}{2}" in label
    assert "(kcal/mol)" in label
    assert metadata["kl_display_equation"] == (
        "0.5 * [p_MD * log(p_MD/p_rw) - p_rw * log(p_rw/p_MD)]"
    )
    assert metadata["kl_display_units"] == "kcal/mol"
    assert metadata["kl_display_cells_zeroed"] == 2
