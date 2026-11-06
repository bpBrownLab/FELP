from __future__ import annotations

import json
from pathlib import Path

from felp.core import MicrostateConfig
from felp.reweighting import MicrostateAtlas


def test_grid_microstate_assignment() -> None:
    atlas = MicrostateAtlas.from_config(
        MicrostateConfig(source="grid", phi_bins=2, psi_bins=2)
    )

    assignment = atlas.assign(-90.0, 90.0)
    assert assignment.state_id == "0_1"
    assert assignment.label == "phi0_psi1"


def test_file_microstate_assignment_wraps_angles(tmp_path: Path) -> None:
    definitions_path = tmp_path / "microstates.json"
    definitions_path.write_text(
        json.dumps(
            {
                "states": [
                    {
                        "state_id": "wrap",
                        "label": "wrap",
                        "phi_range": [150.0, -150.0],
                        "psi_range": [-45.0, 45.0],
                        "reference_probability": 0.75,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    atlas = MicrostateAtlas.from_config(
        MicrostateConfig(source="file", definition_file=str(definitions_path))
    )
    assignment = atlas.assign(179.0, 0.0)
    assert assignment.state_id == "wrap"
    assert atlas.definitions[0].reference_probability == 0.75


def test_file_microstate_assignment_uses_numeric_intervals_for_distance_cvs(
    tmp_path: Path,
) -> None:
    definitions_path = tmp_path / "distance_microstates.json"
    definitions_path.write_text(
        json.dumps(
            {
                "states": [
                    {
                        "state_id": "short",
                        "label": "short",
                        "cv_ranges": {"ends": [0.0, 10.0], "turn": [0.0, 5.0]},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    atlas = MicrostateAtlas.from_config(
        MicrostateConfig(source="file", definition_file=str(definitions_path))
    )

    assignment = atlas.assign({"ends": 8.0, "turn": 4.0})
    assert assignment.state_id == "short"
    assert assignment.cv_centers == {"ends": 5.0, "turn": 2.5}
    assert assignment.phi_center is None
    assert assignment.psi_center is None
