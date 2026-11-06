"""Replay H on all 60 MD-validated directions in the resubmitted paper."""
import gzip
import json
from pathlib import Path

import pytest

from felp.diagnostics.harmonic import harmonic_exposure

DATA = Path(__file__).parent / "data/manuscript_harmonic_cases.json.gz"
with gzip.open(DATA, "rt") as handle:
    CASES = json.load(handle)["cases"]


@pytest.mark.parametrize("case", CASES, ids=[case["case"] for case in CASES])
def test_published_harmonic_exposure(case):
    result = harmonic_exposure(case["state_counts"], case["q"], case["grid_shape"],
                               case["periodic"], case["steps"])
    assert result["H"] == pytest.approx(case["expected_H"], abs=2e-12, rel=0)
    assert result["stationarity_residual"] < 1e-10


def test_fixture_covers_the_manuscript_validation_panel():
    assert len(CASES) == 60
    assert sum("_to_" in c["case"] and c["case"].split("_to_")[0] in {"GLY", "ALA", "PRO"}
               for c in CASES) == 57
    assert {c["case"] for c in CASES if c["case"].startswith("t4l")} == {"t4l_L99A_to_WT"}
