"""Global trimming must match public ArviZ fits and preserve the reference."""
from dataclasses import replace
import numpy as np
import pytest

az = pytest.importorskip('arviz')
from felp.core import ReweightingQCConfig
from felp.reweighting.global_gpd import first_passing_global_gpd_cut, sorted_score_gpd_k
from felp.reweighting.frame_qc import apply_global_frame_qc
from tests.factories import _frame


@pytest.mark.parametrize('kind', ['random', 'ties', 'extreme'])
def test_tail_only_fit_matches_arviz_for_many_prefixes(kind):
    rng = np.random.default_rng(815)
    values = rng.normal(size=340)
    if kind == 'ties':
        values = np.round(values, 1)
    if kind == 'extreme':
        values[:10] -= np.arange(10) * 1000
    ordered = np.sort(values)
    beta = .38
    for start in [0, 1, 4, 10, 20, 100, 280, 310, 320]:
        retained = ordered[start:]
        _, expected = az.psislw(-beta * (retained - retained.min()))
        actual, _ = sorted_score_gpd_k(ordered, start, beta)
        if np.isfinite(expected):
            assert actual == pytest.approx(expected, abs=2e-12)
        else:
            assert actual is None


def test_first_pass_is_strict_and_does_not_assume_monotonicity(monkeypatch):
    import felp.reweighting.global_gpd as module
    shapes = [.7, .5, .9, .49, .2]
    monkeypatch.setattr(module, 'sorted_score_gpd_k', lambda d, i, b: (shapes[i], 10))
    keep, removed, history, reason = module.first_passing_global_gpd_cut(np.arange(50.), beta=1.)
    assert removed.tolist() == [0, 1, 2]
    assert keep.sum() == 47
    assert reason == 'global_gpd_k_passed'
    assert history[-1]['minimum_prior_finite_k'] == .5


def test_unavailable_tail_never_passes_or_deletes_whole_constant_sample():
    keep, removed, _, reason = first_passing_global_gpd_cut(np.zeros(60), beta=1.)
    assert keep.all() and not len(removed)
    assert reason == 'global_gpd_k_unresolved'


def test_global_cut_matches_brute_force_and_is_partition_independent():
    rng = np.random.default_rng(991)
    delta = np.r_[[-40., -30., -20.], rng.normal(size=400)]
    beta = .4
    order = np.argsort(delta, kind='stable')
    expected = None
    for count in range(len(delta) - 19):
        remaining = delta[order[count:]]
        _, k = az.psislw(-beta * (remaining - remaining.min()))
        if np.isfinite(k) and k < .5:
            expected = count
            break
    assert expected is not None and expected > 0
    config = ReweightingQCConfig(policy='exclude-frames', method='gpd',
        exclusion_scope='global', exclusion_strategy='global-gpd-k', pareto_k_threshold=.5)
    frames = [_frame(i, str(i % 2), float(v)) for i, v in enumerate(delta)]
    first = apply_global_frame_qc(frames, qc_config=config, beta=beta)
    second = apply_global_frame_qc([replace(f, microstate_id=str(i % 7)) for i, f in enumerate(frames)],
                                   qc_config=config, beta=beta)
    assert first.excluded_frame_indices == tuple(order[:expected]) == second.excluded_frame_indices
    assert first.final_pareto_k < .5
    assert not first.reliability_unresolved
    assert first.global_ess >= 1


def test_global_reporting_does_not_substitute_local_k_gate():
    import json
    import pandas as pd
    from scripts.manuscript_postprocess_analysis import reporting_mask
    table = pd.DataFrame({'metadata': [json.dumps({'pareto_k': 2., 'state_qc_reasons': ['local tail'],
        'reweighting_qc': {'global_qc': {'final_pareto_k': .49}}})]})
    assert reporting_mask(table, {'qc_reporting_scope': 'global', 'qc_k_threshold': .5}).tolist() == [True]
    with pytest.raises(ValueError, match='local k does not mask states'):
        reporting_mask(table, {'qc_reporting_scope': 'state', 'qc_k_threshold': .5})
