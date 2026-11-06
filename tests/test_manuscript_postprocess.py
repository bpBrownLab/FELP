"""Exercise the manuscript runner on saved scores, including provenance reuse."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('arviz')

from felp.core import OutputConfig
from felp.io import ResultsWriter
from scripts.manuscript_postprocess import build_plan, execute, parse_args
from scripts.manuscript_postprocess_analysis import reporting_mask
from tests.factories import _frame


def test_reporting_requires_available_k_strict_cutoff_and_clear_qc():
    table = pd.DataFrame({'metadata':[json.dumps(m) for m in [
        {'reweighting_qc': {'global_qc': {'final_pareto_k': .49}}},
        {'pareto_k': 2., 'state_qc_reasons': ['local tail']}]]})
    assert reporting_mask(table, {'qc_k_threshold':.5}).tolist() == [True, True]
    for k in (.5, None):
        unavailable = pd.DataFrame({'metadata': [json.dumps({'reweighting_qc': {'global_qc': {'final_pareto_k': k}}})]})
        assert reporting_mask(unavailable, {'qc_k_threshold': .5}).tolist() == [False]


def test_saved_score_pipeline_and_resume(tmp_path):
    writer = ResultsWriter(OutputConfig(output_dir=str(tmp_path/'source')))
    writer.prepare_output_dir()
    rng = np.random.default_rng(327)
    frames = [replace(_frame(i, 'old', float(v)), phi_degrees=-90. if i<100 else 90.)
              for i,v in enumerate(rng.normal(size=200))]
    frames[0] = replace(frames[0], delta_u_kj_mol=-40., perturbed_energy_kj_mol=60.)
    writer.write_frame_results(frames)
    source = Path(writer.raw_frame_results_path)
    before = source.read_bytes()
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'schema_version':1,'runs':[{
        'id':'tiny','group':'test','input_csv':str(source),
        'states':{'source':'grid','phi_bins':2,'psi_bins':1},
        'periodic_cvs':['phi'], 'temperature_kelvin':300., 'prior_mode':'raw_counts',
        'expected_frames':200,'axis_labels':['phi']}]}))
    argv = ['--manifest',str(manifest),'--project-root',str(tmp_path),
            '--output-root',str(tmp_path/'outputs'),'--tag','smoke',
            '--qc-policy','exclude-frames','--tau','100']
    args = parse_args(argv)
    plan = build_plan(args)
    execute(plan, args)
    dest = Path(plan['runs'][0]['output_dir'])
    assert source.read_bytes() == before
    assert not list(dest.rglob('frame_results.csv'))
    assert list((dest/'figures').glob('*.pdf'))
    diag = pd.read_csv(dest/'diagnostics.csv').set_index('method')
    assert diag.loc['neighbor_pool','selected_tau'] == 100
    assert diag.loc['neighbor_pool','weight_treatment'] == 'gpd-psis'
    assert diag.loc['neighbor_pool','reported_target_mass'] <= 1 + 1e-12
    assert diag.loc['independent','excluded_frames'] > 0
    states = pd.read_csv(dest/'state_diagnostics.csv')
    raw_k = states[states.method == 'raw'].set_index('state_id').original_k
    treated = states[states.method == 'independent'].set_index('state_id')
    np.testing.assert_allclose(treated.original_k, raw_k, equal_nan=True)
    assert np.any(np.abs(treated.original_k-treated.retained_raw_k) > .1)
    summaries = pd.read_csv(dest/'treated/reweighting_independent/microstate_summary.csv')
    global_meta = [json.loads(m)['reweighting_qc']['global_qc'] for m in summaries.metadata]
    assert global_meta[0]['excluded_frame_indices']
    assert global_meta[0]['exclusion_history']
    assert 'exclusion_history' not in global_meta[1]
    full = json.loads((dest/'treated/reweighting_independent/run_summary.json').read_text())
    assert full['reweighting_qc']['global_qc']['exclusion_history'] == global_meta[0]['exclusion_history']
    old_time = (dest/'treated/run_summary.json').stat().st_mtime_ns
    execute(plan, parse_args(argv+['--resume']))
    assert (dest/'treated/run_summary.json').stat().st_mtime_ns == old_time
    source.write_bytes(before+b'\n')
    with pytest.raises(ValueError, match='inputs, code, or settings changed'):
        execute(plan, parse_args(argv+['--resume']))


def test_default_commands_use_global_filtering_and_actual_atlas_dimensions(tmp_path):
    from felp.cli import build_parser, build_reweighting_config, build_reweighting_qc_config
    from scripts.manuscript_postprocess import summarize_command

    args = parse_args(['--project-root', str(tmp_path), '--output-root', str(tmp_path/'out'), '--tag', 'paper'])
    run = {'input_csv': 'scores.csv', 'temperature_kelvin': 310., 'prior_mode': 'raw_counts',
           'states': {'source': 'grid', 'phi_bins': 5, 'psi_bins': 5}, 'axis_labels': ['misleading']}
    command = summarize_command(args, run, tmp_path/'out', raw=False)
    namespace = build_parser().parse_args(command[2:])
    qc, pool = build_reweighting_qc_config(namespace), build_reweighting_config(namespace).neighbor_pool
    assert qc.policy == 'exclude-frames' and qc.exclusion_strategy == 'global-gpd-k'
    assert qc.pareto_k_threshold == .5 and qc.weight_treatment == 'gpd-psis'
    assert pool.grid_connectivity == 'chebyshev' and pool.chebyshev_radius == 3
    atlas = tmp_path/'atlas.json'
    atlas.write_text(json.dumps({'states': [
        {'state_id': str(i), 'label': str(i), 'cv_ranges': {'x': [i, i+1]}} for i in range(3)
    ]}))
    run['states'] = {'source': 'file', 'file': str(atlas)}
    command = summarize_command(args, run, tmp_path/'out', raw=False)
    assert build_reweighting_config(build_parser().parse_args(command[2:])).neighbor_pool.grid_connectivity == 'cardinal'
