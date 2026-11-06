"""Analytic Dirichlet solutions and original-support semantics of manuscript H."""
import json
from dataclasses import replace

import numpy as np
import pytest

from felp.core import MicrostateDefinition, ReweightingConfig, ReweightingQCConfig
from felp.diagnostics.harmonic import (
    diagnose_payload, grid_from_atlas, harmonic_exposure, harmonic_exposure_for_atlas,
)
from felp.reweighting import MicrostateAtlas, ReweightingOrchestrator
from tests.factories import _frame


def line_atlas(n=5):
    return MicrostateAtlas.from_definitions([
        MicrostateDefinition(state_id=f"s{i}", label=f"s{i}", cv_ranges={"x": (float(i), float(i+1))})
        for i in range(n)
    ], reference_probability_mode="raw_counts")


def test_analytic_chain_and_tv_identity():
    q = np.array([.36, 0., .64])
    result = harmonic_exposure([3, 0, 4], q, (3,), (False,), (1.,))
    continued = np.array([.36, .49, .64]) / 1.49
    assert result["E"] == pytest.approx(.49)
    assert result["H"] == pytest.approx(.5 * np.abs(continued - q).sum())
    assert result["stationarity_residual"] < 1e-12
    assert result["physical_missing_mass_estimate"] is False


def test_zero_target_probability_is_still_a_visited_anchor():
    result = harmonic_exposure([3, 0, 1, 0, 4], [.36, 0, 0, 0, .64],
                               (5,), (False,), (1.,))
    assert result["H"] == pytest.approx(.2)
    assert result["unvisited_state_count"] == 2


def test_periodic_translation_and_common_unit_scale():
    q = np.array([[.2, 0, 0], [0, .3, 0], [0, 0, .5]])
    counts = (q > 0).astype(int)
    expected = harmonic_exposure(counts.ravel(), q.ravel(), (3, 3), (True, True), (2., 5.))
    for shift, scale in [(1, 1e-3), (2, 1e3)]:
        actual = harmonic_exposure(np.roll(counts, shift, axis=0).ravel(),
                                   np.roll(q, shift, axis=0).ravel(),
                                   (3, 3), (True, True), (2*scale, 5*scale))
        assert actual["H"] == pytest.approx(expected["H"])


def test_full_support_is_zero_even_when_local_ess_is_small():
    result = diagnose_payload({"schema_version": 1, "state_ids": ["a", "b"],
        "state_counts": [1, 2], "q": [.2, .8], "grid_shape": [2],
        "steps": [1.], "periodic": [False], "local_ess": [1., 2.]})
    assert result["H"] == 0.
    assert result["U"] == 1.
    assert result["status"] == "no_unvisited_cells"


def test_atlas_order_comes_from_physical_intervals():
    atlas = line_atlas(3)
    shuffled = replace(atlas, definitions=tuple(reversed(atlas.definitions)))
    assert grid_from_atlas(shuffled).state_ids == ("s0", "s1", "s2")
    result = harmonic_exposure_for_atlas(shuffled, {"s0": 3, "s2": 4}, {"s0": .36, "s2": .64})
    assert result["H"] == pytest.approx(.49 / 1.49)
    sparse = replace(atlas, definitions=(atlas.definitions[0], atlas.definitions[2]))
    result = harmonic_exposure_for_atlas(sparse, {"s0": 3, "s2": 4}, {"s0": .36, "s2": .64})
    assert result["H"] is None
    assert result["status"] == "unavailable_atlas"


def test_failed_score_does_not_erase_a_visited_conformation():
    frames = [_frame(i, "s0" if i < 3 else "s4", 0.) for i in range(7)]
    frames.append(replace(_frame(7, "s2", 0.), metadata={"status": "failed"}))
    outcome = ReweightingOrchestrator(300., ReweightingConfig(methods=("independent",)),
        qc_config=ReweightingQCConfig(policy="none")).run_single(frames, line_atlas())
    diagnostic = outcome.shared_metadata["harmonic_exposure"]
    assert diagnostic["H"] == pytest.approx(.2)
    assert diagnostic["unvisited_state_count"] == 2
    assert outcome.shared_metadata["manuscript_diagnostics"]["U"] == pytest.approx(1.)


@pytest.mark.parametrize("field,value", [
    ("q", [.4, .1, .5]), ("state_counts", [1, -.1, 1]),
    ("grid_shape", [True]), ("periodic", ["false"]),
    ("steps", [0.]), ("local_ess", [1., float("nan"), 2.]),
    ("target_md", [.2, .3, .5]),
])
def test_invalid_or_target_dependent_inputs_fail(field, value):
    payload = {"schema_version": 1, "state_ids": ["a", "b", "c"],
        "state_counts": [1, 0, 1], "q": [.4, 0, .6], "grid_shape": [3],
        "periodic": [False], "steps": [1.]}
    payload[field] = value
    with pytest.raises(ValueError):
        diagnose_payload(payload)


def test_cli_writes_separate_diagnostics_and_refuses_overwrite(tmp_path, capsys):
    from felp.app import main
    source = tmp_path / "input.json"
    output = tmp_path / "diagnostic.json"
    source.write_text(json.dumps({"schema_version": 1, "state_ids": ["a", "b", "c"],
        "state_counts": [1, 0, 1], "q": [.36, 0, .64], "grid_shape": [3],
        "periodic": [False], "steps": [1.]}))
    assert main(["harmonic-exposure", "--input", str(source), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["H"] == pytest.approx(.49 / 1.49)
    with pytest.raises(FileExistsError):
        main(["harmonic-exposure", "--input", str(source), "--output", str(output)])
