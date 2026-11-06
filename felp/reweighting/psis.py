"""Standard generalized-Pareto PSIS with the input weight scale retained.

ArviZ returns normalized log weights. FELP needs unnormalized local means:
normalizing each state's weights separately would erase relative state factors.
An unchanged finite bulk weight recovers ArviZ's removed normalizing constant.
"""
from __future__ import annotations

import warnings
import numpy as np

from felp.core import ReweightingError


def gpd_psis_log_weights(log_weights: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    """Smooth using ArviZ PSIS (nominal reff=1), preserving inter-state scale.

    Zero weights (log weight -inf) remain zero. NaN and +inf are errors.
    Insufficient strict tails are unavailable fits, not evidence of good overlap.
"""
    try:
        import arviz as az
    except ImportError as exc:
        raise ReweightingError(
            "gpd-psis requires ArviZ; install scripts/requirements_postprocess.txt."
        ) from exc
    if az.__version__ != "0.22.0":
        raise ReweightingError(
            "gpd-psis is validated for ArviZ 0.22.0; install scripts/requirements_postprocess.txt."
        )
    lw = np.asarray(log_weights, dtype=float)
    if lw.ndim != 1 or np.any(np.isnan(lw)) or np.any(np.isposinf(lw)):
        raise ReweightingError("GPD PSIS requires 1D log weights without NaN or +inf.")
    meta = dict(psis_estimator="arviz_gpd", psis_version=az.__version__,
                psis_reff=1.0, psis_smoothed=False, psis_smoothed_frame_count=0,
                psis_pareto_k=None, psis_tail_count=0, psis_truncation_cap=None,
                psis_tail_cap_rule="observed maximum input log weight",
                psis_log_truncation_cap=float(np.max(lw)) if np.any(np.isfinite(lw)) else None,
                psis_fit_status="insufficient_strict_tail")
    if len(lw) < 6 or not np.any(np.isfinite(lw)):
        return lw.copy(), meta
    positive = np.isfinite(lw)
    relative = lw - np.max(lw)
    cutoff_index = -int(np.ceil(min(len(lw)/5, 3*np.sqrt(len(lw))))) - 1
    cutoff = max(float(np.sort(relative)[cutoff_index]), np.log(np.finfo(float).tiny))
    tail = relative > cutoff
    meta['psis_tail_count'] = int(tail.sum())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', RuntimeWarning)
        smoothed_lw, k = az.psislw(lw, reff=1.0)
    meta['psis_runtime_warnings'] = [str(x.message) for x in caught]
    k = float(k)
    meta['psis_pareto_k'] = k if np.isfinite(k) else None
    meta['psis_fit_status'] = ('finite' if np.isfinite(k) else
                              'insufficient_strict_tail' if tail.sum() <= 4 else 'nonfinite_fit')
    if not np.isfinite(k):
        return lw.copy(), meta
    bulk = positive & ~tail
    if not bulk.any():
        raise ReweightingError("No finite bulk weight to restore PSIS scale; supply original log scores before underflow.")
    anchor = np.flatnonzero(bulk)[0]
    restored = smoothed_lw + (lw[anchor] - smoothed_lw[anchor])
    # Preserve untouched bulk weights exactly, including their absolute scale.
    np.testing.assert_allclose(restored[bulk] - lw[bulk], 0, atol=1e-9)
    restored[~tail] = lw[~tail]
    meta['psis_smoothed'] = bool(np.isfinite(k))
    meta['psis_smoothed_frame_count'] = int(tail.sum()) if np.isfinite(k) else 0
    return restored, meta


def gpd_psis_weights(weights: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    """Weight-input adapter; score consumers should use the log-input function."""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ReweightingError("GPD PSIS requires finite, nonnegative 1D weights.")
    lw = np.full(len(w), -np.inf)
    positive = w > 0
    lw[positive] = np.log(w[positive])
    result, meta = gpd_psis_log_weights(lw)
    return np.exp(result), meta


def treat_weights(weights: np.ndarray, treatment: str, *, log_weights: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, object]]:
    """Apply raw weights or the ArviZ PSIS treatment."""
    if treatment == 'raw':
        return np.asarray(weights, dtype=float).copy(), {}
    if treatment == 'gpd-psis':
        if log_weights is not None:
            treated, meta = gpd_psis_log_weights(log_weights)
            return np.exp(treated), meta
        return gpd_psis_weights(weights)
    raise ReweightingError(f"Unknown weight treatment: {treatment}")
