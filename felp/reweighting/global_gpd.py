"""Explicit global GPD-k trimming of a fixed ordered score sample.

This defines a trimmed estimator. Passing describes the retained raw weights,
not the original estimator or the physical accuracy of the scores.
"""
from __future__ import annotations

import warnings
import numpy as np


def sorted_score_gpd_k(ordered_delta, start, beta):
    """Fit ArviZ 0.22's reff=1 tail from ascending scores.

    Removing a score prefix removes the largest weights. Only the candidate
    tail is materialized, avoiding another sort of the full sample.
    """
    import arviz
    from arviz.stats.stats import _gpdfit
    if arviz.__version__ != '0.22.0':
        raise RuntimeError('Global GPD trimming is validated for ArviZ 0.22.0')
    count = len(ordered_delta) - start
    if count < 6:
        return None, 0
    tail_count = int(np.ceil(min(count / 5., 3. * np.sqrt(count))))
    minimum = ordered_delta[start]
    cutoff = max(float(-beta * (ordered_delta[start + tail_count] - minimum)),
                 float(np.log(np.finfo(float).tiny)))
    lw = -beta * (ordered_delta[start:start + tail_count][::-1] - minimum)
    lw = lw[lw > cutoff]
    if len(lw) <= 4:
        return None, len(lw)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        k, _ = _gpdfit(np.exp(lw) - np.exp(cutoff))
    return (float(k) if np.isfinite(k) else None), len(lw)


def first_passing_global_gpd_cut(delta, *, beta, cutoff=.5, min_retained=20):
    """Return the first passing cut, retained mask and diagnostic history.

    k need not change monotonically, so every prefix is evaluated. Record every
    100th fit and the crossing. If none passes before the minimum retained count,
    return the original sample with unresolved status.
    """
    delta = np.asarray(delta, dtype=float)
    if delta.ndim != 1 or not np.all(np.isfinite(delta)):
        raise ValueError('Global GPD trimming requires finite one-dimensional scores')
    if not np.isfinite(beta) or beta <= 0 or not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError('beta and cutoff must be finite and positive')
    if min_retained < 6:
        raise ValueError('min_retained must be at least six')
    order = np.argsort(delta, kind='stable')
    ordered = delta[order]
    history = []
    # Every deletion prefix of a constant vector still has an empty strict
    # tail. Preserve the unresolved result without refitting up to N times.
    # In particular, a zero-energy identity control is never a passing fit.
    if len(delta) and ordered[0] == ordered[-1]:
        history.append({'excluded_frame_count': 0, 'retained_frame_count': len(delta),
                        'pareto_k': None, 'tail_count': 0, 'cutoff': cutoff,
                        'fit_status': 'constant_weights',
                        'reason': 'Every deletion prefix has an empty strict tail'})
        return np.ones(len(delta), dtype=bool), order[:0], history, 'global_gpd_k_unresolved'
    previous = None
    minimum_prior_k = None
    for removed in range(max(0, len(delta) - min_retained + 1)):
        k, tail = sorted_score_gpd_k(ordered, removed, beta)
        current = {'excluded_frame_count': removed, 'retained_frame_count': len(delta) - removed,
                   'pareto_k': k, 'tail_count': tail, 'cutoff': cutoff,
                   'fit_status': 'finite' if k is not None else 'unavailable'}
        if removed % 100 == 0:
            history.append(current)
        if k is not None and k < cutoff:
            if previous is not None and previous not in history:
                history.append(previous)
            if current not in history:
                history.append(current)
            current.update(prefixes_evaluated=removed + 1, minimum_prior_finite_k=minimum_prior_k)
            keep = np.ones(len(delta), dtype=bool)
            keep[order[:removed]] = False
            return keep, order[:removed], history, 'global_gpd_k_passed'
        if k is not None:
            minimum_prior_k = k if minimum_prior_k is None else min(minimum_prior_k, k)
        previous = current
    if previous is not None and previous not in history:
        history.append(previous)
    return np.ones(len(delta), dtype=bool), order[:0], history, 'global_gpd_k_unresolved'
