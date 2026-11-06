"""Exercise the actual shared-sample six-way matrix on saved score data."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

pytest.importorskip('arviz')
from felp.core import OutputConfig
from felp.io.results import ResultsWriter
from scripts.manuscript_uniform_matrix import PROTOCOL, code_identity, run_system
from tests.factories import _frame


def test_prepare_is_portable_and_selects_requested_cases(tmp_path):
    from scripts.manuscript_uniform_matrix import prepare
    import sys

    (tmp_path / 'scores.csv').write_text('frame_index\n')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'schema_version': 1, 'runs': [{
        'id': name, 'group': 'test', 'input_csv': 'scores.csv',
        'states': {'source': 'grid', 'phi_bins': 3, 'psi_bins': 3},
        'temperature_kelvin': 310., 'prior_mode': 'raw_counts', 'periodic_cvs': ['phi', 'psi'],
    } for name in ['first', 'second']]}))
    base = tmp_path / 'matrix'
    prepare(base, tmp_path, 1, manifest=manifest, systems=['second'])
    plan = json.loads((base / 'control/matrix_plan.json').read_text())
    assert len(plan['tasks']) == 1
    assert plan['tasks'][0]['id'] == 'second'
    assert plan['tasks'][0]['host'] == 'local'
    assert plan['tasks'][0]['command'][0] == sys.executable
    assert 'LD_LIBRARY_PATH' not in plan['environment']


def test_all_six_settings_share_one_global_cut_and_original_prior(tmp_path):
    rng = np.random.default_rng(293)
    delta = rng.normal(size=480)
    delta[:3] = [-50., -35., -22.]
    frames = [replace(_frame(i, 'old', float(d)), phi_degrees=-157.5 + 45 * (i % 8)) for i, d in enumerate(delta)]
    writer = ResultsWriter(OutputConfig(output_dir=str(tmp_path / 'input')))
    writer.prepare_output_dir()
    writer.write_frame_results(frames)
    source = Path(writer.raw_frame_results_path)
    original = source.read_bytes()
    root = tmp_path / 'results'
    run = {'id': 'fixture', 'group': 'test', 'input_csv': str(source), 'output_dir': str(root),
           'states': {'source': 'grid', 'phi_bins': 8, 'psi_bins': 1},
           'periodic_cvs': ['phi'], 'temperature_kelvin': 310., 'expected_frames': len(frames),
           'prior_mode': 'raw_counts', 'axis_labels': ['phi'], 'target_probability_csv': None}
    spec = {'run': run, 'protocol': PROTOCOL, 'code': code_identity(), 'workers': 2}
    result = run_system(spec)
    assert result['status'] == 'complete' and result['global_k_pass']
    assert result['excluded_frames'] > 0
    assert set(result['comparisons']) == set(PROTOCOL['comparisons'])
    hashes = {r['excluded_ids_sha256'] for r in result['comparisons'].values()}
    assert len(hashes) == 1
    for name, record in result['comparisons'].items():
        if 'tau1' == name.split('_')[1]:
            assert record['selected_tau'] == 1
        if 'tau100' == name.split('_')[1]:
            assert record['selected_tau'] == 100
        assert record['global_k'] < .5
        table = pd.read_csv(Path(record['output_dir']) / 'treated/reweighting_neighbor_pool/microstate_summary.csv')
        np.testing.assert_allclose(table.reference_probability, np.full(8, .125))
        assert table.target_probability.sum() == pytest.approx(1)
        diag = pd.read_csv(Path(record['output_dir']) / 'diagnostics.csv').set_index('method')
        assert 'raw_eta' not in diag
        assert 'warning_raw_eta_processed_demand' not in diag
        assert diag.loc['neighbor_pool', 'H'] == pytest.approx(0.)
        assert diag.loc['neighbor_pool', 'U'] == pytest.approx(diag.loc['neighbor_pool', 'predicted_mass_local_ess_lt10'])
        assert diag.loc['neighbor_pool', 'reporting_k_scope'] == 'global'
        assert diag.loc['neighbor_pool', 'reported_target_mass'] == pytest.approx(1)
        assert list((Path(record['output_dir']) / 'figures').glob('*.pdf'))
    assert source.read_bytes() == original
    assert not list(root.rglob('frame_results.csv'))
    modified = (root / 'filtered_psis/microstate_summary.csv').stat().st_mtime_ns
    run_system(spec, resume=True)
    assert (root / 'filtered_psis/microstate_summary.csv').stat().st_mtime_ns == modified
    source.write_bytes(original + b'\n')
    with pytest.raises(ValueError, match='incomplete or changed'):
        run_system(spec, resume=True)
