"""Reference-implementation and estimator-scale regressions for ArviZ PSIS."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.special import logsumexp

az = pytest.importorskip('arviz')

from felp.core import ConfigurationError, ReweightingQCConfig, frame_reweighting_delta
from felp.reweighting.aggregator import _state_qc_report
from felp.reweighting.psis import gpd_psis_log_weights, gpd_psis_weights, treat_weights
from felp.reweighting.prep import build_reweighting_prep
from felp.plotting.frame_projection_weights import treated_weights_for_frames
from tests.factories import _frame, _atlas, _bulk_with_outliers, BETA


def test_arviz_normalization_bulk_and_interstate_scale():
    lw = np.log(np.random.default_rng(732).pareto(1.4, 1000) + 1)
    original = lw.copy()
    actual, meta = gpd_psis_log_weights(lw)
    expected, k = az.psislw(lw, reff=1.)
    np.testing.assert_array_equal(lw, original)
    np.testing.assert_allclose(actual-logsumexp(actual), expected, atol=1e-12)
    assert meta['psis_pareto_k'] == pytest.approx(k)
    assert meta['psis_estimator'] == 'arviz_gpd'
    assert meta['psis_tail_count'] > 4
    bulk = lw <= np.sort(lw)[-meta['psis_tail_count']-1]
    np.testing.assert_array_equal(actual[bulk], lw[bulk])
    assert actual.max() <= lw.max() + 1e-12
    for shift in [-900., 600.]:
        shifted, _ = gpd_psis_log_weights(lw+shift)
        np.testing.assert_allclose(shifted-shift, actual, atol=1e-10)


@pytest.mark.parametrize('weights', [np.zeros(50), np.ones(50), np.arange(1., 6.)])
def test_unavailable_fits_preserve_weights_without_claiming_success(weights):
    result, meta = gpd_psis_weights(weights)
    np.testing.assert_allclose(result, weights)
    assert not meta['psis_smoothed']
    assert meta['psis_pareto_k'] is None


def test_original_log_scores_preserve_bulk_before_underflow():
    lw = np.r_[np.full(200, -800.), np.linspace(-10, 0, 40)]
    actual, meta = treat_weights(np.exp(lw), 'gpd-psis', log_weights=lw)
    expected, k = az.psislw(lw)
    np.testing.assert_allclose(actual/actual.sum(), np.exp(expected), atol=1e-12)
    assert meta['psis_pareto_k'] == pytest.approx(k)


@pytest.mark.parametrize('lw', [[np.nan], [np.inf], [[0., 1.]]])
def test_bad_log_weights_are_rejected(lw):
    with pytest.raises(Exception, match='1D log weights'):
        gpd_psis_log_weights(lw)




def test_strict_fixed_k_cutoff_and_sidechain_aware_diagnostic():
    values = np.random.default_rng(6).normal(size=300)
    frames = [replace(_frame(i, 's0', 0.), reweighting_delta_kj_mol=v)
              for i, v in enumerate(values)]
    qc = ReweightingQCConfig(method='gpd', pareto_k_threshold=.5)
    report = _state_qc_report(frames, beta=BETA, config=qc)
    _, expected_k = az.psislw(-BETA*(values-values.min()))
    assert report.pareto_k == pytest.approx(expected_k)
    assert report.pareto_estimator == 'arviz_gpd'
    # Equality must fail the requested strict reporting rule.
    report = _state_qc_report(frames, beta=BETA,
                              config=replace(qc, pareto_k_threshold=float(expected_k)))
    assert report.status == 'flagged'


@pytest.mark.parametrize('cutoff', [0, -1, np.nan, np.inf])
def test_invalid_cutoff_rejected(cutoff):
    with pytest.raises(ConfigurationError):
        ReweightingQCConfig(pareto_k_threshold=cutoff)


def test_global_qc_precedes_psis_and_projection_matches_state_means():
    frames = _bulk_with_outliers(n_bulk=150, outliers=(-40.,))
    qc = ReweightingQCConfig(policy='exclude-frames', method='gpd',
                            weight_treatment='gpd-psis', exclusion_scope='global',
                            pareto_k_threshold=.5)
    prep = build_reweighting_prep(frames, _atlas(['s0']), beta=BETA,
                                 temperature_kelvin=300., qc_config=qc)
    assert prep.global_qc.excluded_frame_indices == (1000,)
    state = prep.state_data['s0']
    values = np.array([frame_reweighting_delta(f) for f in state.frames])
    _, expected_k = az.psislw(-BETA*(values-values.min()))
    assert state.psis_weight_metadata['psis_pareto_k'] == pytest.approx(expected_k)
    projected = treated_weights_for_frames(state.frames, beta=BETA,
                    global_delta_shift=prep.global_delta_shift, weight_treatment='gpd-psis')
    np.testing.assert_allclose(projected, state.treated_weights)
    assert projected.mean() == pytest.approx(state.local_factor)
