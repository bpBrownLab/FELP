"""Public options reject retired algorithms instead of silently substituting one."""
import pytest
from felp.cli import build_parser
from felp.core import ConfigurationError, NeighborPoolConfig, ReweightingConfig, ReweightingQCConfig


@pytest.mark.parametrize('method', ['cumulant', 'laplacian', 'frame'])
def test_retired_estimators_are_rejected_in_configs_and_cli(method):
    with pytest.raises(ConfigurationError, match='Unsupported reweighting method'):
        ReweightingConfig.from_dict({'methods': [method]})
    with pytest.raises(SystemExit):
        build_parser().parse_args(['summarize', '--raw-frame-results', 'scores.csv',
                                  '--output-dir', 'out', '--reweighting-method', method])


@pytest.mark.parametrize('options', [
    {'solver': 'variational'}, {'tau_mode': 'gcv'}, {'tau_mode': 'mean_ess'},
    {'tau_mode': 'median_ess'}, {'min_local_weight': .1}, {'tau': float('nan')},
])
def test_unsupported_pooling_options_fail(options):
    with pytest.raises(ConfigurationError):
        NeighborPoolConfig.from_dict(options)


@pytest.mark.parametrize('options', [
    {'method': 'threshold'}, {'method': 'auto'}, {'weight_treatment': 'psis-smoothed'},
    {'weight_treatment': 'psis'}, {'exclusion_scope': 'state'},
    {'exclusion_strategy': 'robust-chauvenet'}, {'max_excluded_frames_per_state': 5},
])
def test_retired_filtering_and_smoothing_options_fail(options):
    with pytest.raises(ConfigurationError):
        ReweightingQCConfig.from_dict(options)


@pytest.mark.parametrize('command', ['reportability', 'ess-compare', 'amber-score'])
def test_retired_commands_are_unavailable(command):
    with pytest.raises(SystemExit):
        build_parser().parse_args([command])
