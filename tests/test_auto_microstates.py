from __future__ import annotations

import numpy as np

from felp.reweighting.auto_microstates import build_auto_microstate_payload


def test_build_auto_microstate_payload_writes_complete_two_cv_grid() -> None:
    ends = np.linspace(8.0, 24.0, 200)
    turn = np.concatenate(
        [
            np.linspace(3.0, 6.0, 100),
            np.linspace(7.0, 12.0, 100),
        ]
    )

    payload = build_auto_microstate_payload(
        cv_values={"ends": ends, "turn": turn},
        cv_names=("ends", "turn"),
    )

    metadata = payload["metadata"]
    states = payload["states"]
    x_bins = metadata["axes"]["ends"]["bin_count"]
    y_bins = metadata["axes"]["turn"]["bin_count"]

    assert metadata["cv_names"] == ["ends", "turn"]
    assert metadata["frame_count"] == 200
    assert len(states) == x_bins * y_bins
    assert len(states) <= metadata["policy"]["max_total_states"]
    assert all(set(state["cv_ranges"]) == {"ends", "turn"} for state in states)
    assert all("plot_x" in state and "plot_y" in state for state in states)


def test_auto_microstate_default_cap_scales_with_large_frame_count() -> None:
    payload = build_auto_microstate_payload(
        cv_values={
            "ends": np.linspace(0.0, 30.0, 25_000),
            "turn": np.linspace(4.0, 11.0, 25_000),
        },
        cv_names=("ends", "turn"),
    )

    metadata = payload["metadata"]

    assert metadata["state_count"] > 144
    assert metadata["state_count"] <= metadata["policy"]["max_total_states"]
    assert metadata["state_count"] <= metadata["policy"]["max_states_by_support"]


def test_build_auto_microstate_payload_pads_constant_axis() -> None:
    payload = build_auto_microstate_payload(
        cv_values={
            "ends": np.full(50, 12.0),
            "turn": np.linspace(2.0, 10.0, 50),
        },
        cv_names=("ends", "turn"),
    )

    ends_edges = payload["metadata"]["axes"]["ends"]["edges"]

    assert payload["metadata"]["axes"]["ends"]["bin_count"] == 1
    assert ends_edges[0] < 12.0 < ends_edges[1]
