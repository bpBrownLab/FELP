"""Published protocol selection and a saved-score CLI run without rescoring."""
import json
from pathlib import Path

import numpy as np
import pytest

from felp.core import ConfigurationError, MicrostateConfig, OutputConfig
from felp.io.results import ResultsWriter
from felp.reweighting import MicrostateAtlas
from felp.reweighting.manuscript import manuscript_configs
from tests.test_harmonic_exposure import line_atlas
from tests.factories import _frame


def test_published_config_and_geometry_selection():
    grid = MicrostateAtlas.from_config(MicrostateConfig(phi_bins=45, psi_bins=45,
        reference_probability_mode="raw_counts"))
    config, qc = manuscript_configs(grid, periodic_cv_names=("phi", "psi"))
    assert config.methods == ("neighbor_pool",)
    assert config.neighbor_pool.grid_connectivity == "chebyshev"
    assert config.neighbor_pool.chebyshev_radius == 3
    assert config.neighbor_pool.periodic_cv_names == ("phi", "psi")
    assert config.neighbor_pool.solver == "blend"
    assert config.neighbor_pool.tau is None
    assert config.neighbor_pool.tau_mode == "lcurve"
    assert (qc.exclusion_strategy, qc.exclusion_scope, qc.pareto_k_threshold,
            qc.weight_treatment, qc.min_frame_count) == ("global-gpd-k", "global", .5, "gpd-psis", 20)
    config, _ = manuscript_configs(line_atlas())
    assert config.neighbor_pool.grid_connectivity == "cardinal"
    with pytest.raises(ConfigurationError, match="Periodic"):
        manuscript_configs(line_atlas(), periodic_cv_names=("phi",))


def test_saved_score_protocol_preserves_prior_and_writes_h_and_u(tmp_path, capsys):
    pytest.importorskip("arviz")
    from felp.app import main

    atlas_file = tmp_path / "atlas.json"
    atlas_file.write_text(json.dumps({"states": [
        {"state_id": f"s{i}", "label": f"s{i}", "cv_ranges": {"x": [i, i+1]}}
        for i in range(3)
    ]}))
    source_dir, dest = tmp_path / "scores", tmp_path / "summarized"
    writer = ResultsWriter(OutputConfig(output_dir=str(source_dir)))
    writer.prepare_output_dir()
    values = np.random.default_rng(991).normal(size=400)
    values[0] = -40.
    writer.write_frame_results([_frame(i, "s0" if i < 100 else "s2", float(v))
                                for i, v in enumerate(values)])
    source = Path(writer.raw_frame_results_path)
    original = source.read_bytes()
    assert main(["summarize", "--protocol", "jctc2026", "--raw-frame-results", str(source),
                 "--output-dir", str(dest), "--microstate-source", "file", "--microstate-file", str(atlas_file),
                 "--prior-probability-mode", "raw_counts", "--temperature-kelvin", "310",
                 # Incomplete individual QC settings are overridden as a unit.
                 "--reweighting-qc-exclusion-strategy", "global-gpd-k"]) == 0
    result = json.loads((dest / "run_summary.json").read_text())
    assert result["processing_protocol"]["id"] == "jctc2026"
    assert result["overlap_assessment"] is None
    diagnostics = result["overlap_diagnostics"]
    assert diagnostics["harmonic_exposure"]["unvisited_state_count"] == 1
    assert 0 < diagnostics["manuscript_diagnostics"]["H"] < 1
    assert result["reweighting_qc"]["global_qc"]["final_pareto_k"] < .5
    assert source.read_bytes() == original
    import pandas as pd
    table = pd.read_csv(dest / "microstate_summary.csv").set_index("state_id")
    np.testing.assert_allclose(table.reference_probability, [.25, 0., .75])
    assert table.loc["s1", "target_probability"] == 0.


def test_constant_identity_keeps_reference_without_falsely_passing_tail():
    pytest.importorskip("arviz")
    from felp.reweighting import ReweightingOrchestrator

    atlas = line_atlas(3)
    config, qc = manuscript_configs(atlas)
    frames = [_frame(i, "s0" if i < 30 else "s2", 0.) for i in range(120)]
    result = ReweightingOrchestrator(310., config, qc).run_single(frames, atlas)
    np.testing.assert_allclose([s.target_probability for s in result.microstate_results], [.25, 0., .75])
    tail = result.shared_metadata["reweighting_qc"]["global_qc"]
    assert tail["final_pareto_k"] is None
    assert not tail["excluded_frame_indices"]
