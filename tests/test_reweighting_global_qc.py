"""Global score filtering preserves the reference prior across partitions."""
from dataclasses import replace
import numpy as np
import pytest

from felp.core import K_BOLTZMANN_KJ_PER_MOL_K, MicrostateDefinition, ReweightingQCConfig
from felp.reweighting import MicrostateAtlas
from felp.reweighting.prep import build_reweighting_prep
from tests.factories import _frame

BETA = 1 / (K_BOLTZMANN_KJ_PER_MOL_K * 310)


def _atlas(size):
    return MicrostateAtlas.from_definitions([
        MicrostateDefinition(state_id=f's{i}', label=f's{i}', cv_ranges={'x': (i, i+1)})
        for i in range(size)
    ], reference_probability_mode='raw_counts')


def test_global_cut_and_original_priors_are_partition_independent():
    scores = np.r_[[-40., -30., -20.], np.random.default_rng(991).normal(size=400)]
    qc = ReweightingQCConfig(policy='exclude-frames', weight_treatment='gpd-psis')
    excluded = []
    for bins in (1, 2, 7):
        frames = [_frame(i, f's{i % bins}', float(v)) for i, v in enumerate(scores)]
        prep = build_reweighting_prep(frames, _atlas(bins), beta=BETA,
                                     temperature_kelvin=310., qc_config=qc)
        excluded.append(prep.global_qc.excluded_frame_indices)
        assert prep.global_qc.final_pareto_k < .5
        assert sum(len(s.frames) for s in prep.state_data.values()) == len(scores)-len(excluded[-1])
        for i, state in enumerate(prep.state_data.values()):
            assert not state.excluded_frame_indices
            assert state.reference_probability == pytest.approx(sum(j % bins == i for j in range(len(scores)))/len(scores))
    assert excluded[0] and excluded[0] == excluded[1] == excluded[2]


def test_filtering_all_scores_in_a_state_preserves_its_visited_status():
    scores = np.random.default_rng(45).normal(size=400)
    frames = [_frame(i, 's0', float(v)) for i, v in enumerate(scores)]
    frames.append(_frame(400, 's1', -100.))
    prep = build_reweighting_prep(frames, _atlas(3), beta=BETA, temperature_kelvin=310.,
                                 qc_config=ReweightingQCConfig(policy='exclude-frames'))
    assert prep.state_data['s1'].frames == []
    assert prep.state_data['s1'].reference_probability == pytest.approx(1/401)
    assert prep.coverage_reference_counts['s1'] == 1
    assert prep.coverage_reference_counts.get('s2', 0) == 0


def test_statewise_psis_follows_the_global_cut_and_preserves_scale():
    from felp.reweighting.psis import gpd_psis_log_weights
    scores = np.r_[[-40., -30.], np.random.default_rng(21).normal(size=400)]
    frames = [_frame(i, f's{i % 2}', float(v)) for i, v in enumerate(scores)]
    prep = build_reweighting_prep(frames, _atlas(2), beta=BETA, temperature_kelvin=310.,
                                 qc_config=ReweightingQCConfig(policy='exclude-frames', weight_treatment='gpd-psis'))
    for state in prep.state_data.values():
        log_weights = -BETA*(np.array([f.delta_u_kj_mol for f in state.frames])-prep.global_delta_shift)
        expected, _ = gpd_psis_log_weights(log_weights)
        np.testing.assert_allclose(state.treated_weights, np.exp(expected), rtol=1e-13)
