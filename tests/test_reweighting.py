from __future__ import annotations

import json
import math
from pathlib import Path

from felp.core import FrameResult, MicrostateDefinition
from felp.reweighting import (
    MicrostateAtlas,
    ReweightingAggregator,
    load_microstate_definitions,
)


def _frame_result(frame_index: int, state_id: str, delta_u: float) -> FrameResult:
    return FrameResult(
        frame_index=frame_index,
        stride_number=frame_index,
        microstate_id=state_id,
        microstate_label=state_id,
        phi_degrees=0.0,
        psi_degrees=0.0,
        reference_energy_kj_mol=0.0,
        perturbed_energy_kj_mol=delta_u,
        delta_u_kj_mol=delta_u,
        reference_relaxed_phi_degrees=0.0,
        reference_relaxed_psi_degrees=0.0,
        perturbed_relaxed_phi_degrees=0.0,
        perturbed_relaxed_psi_degrees=0.0,
    )


def test_reweighting_uses_empirical_reference_probabilities() -> None:
    atlas = MicrostateAtlas(
        definitions=(
            MicrostateDefinition("a", "a", (-180.0, 0.0), (-180.0, 180.0)),
            MicrostateDefinition("b", "b", (0.0, 180.0), (-180.0, 180.0)),
        )
    )
    aggregator = ReweightingAggregator(temperature_kelvin=300.0)
    results = aggregator.aggregate(
        [
            _frame_result(0, "a", 0.0),
            _frame_result(1, "a", 0.0),
            _frame_result(2, "b", 10.0),
        ],
        atlas,
    )

    result_by_id = {result.state_id: result for result in results}
    assert result_by_id["a"].reference_probability == 2.0 / 3.0
    assert result_by_id["b"].reference_probability == 1.0 / 3.0
    assert result_by_id["a"].target_probability > result_by_id["b"].target_probability
    assert abs(sum(result.target_probability for result in results) - 1.0) < 1.0e-8


def test_reweighting_ignores_failed_frames() -> None:
    atlas = MicrostateAtlas(
        definitions=(
            MicrostateDefinition("a", "a", (-180.0, 0.0), (-180.0, 180.0)),
            MicrostateDefinition("b", "b", (0.0, 180.0), (-180.0, 180.0)),
        )
    )
    aggregator = ReweightingAggregator(temperature_kelvin=300.0)

    failed = FrameResult(
        frame_index=99,
        stride_number=99,
        microstate_id="failed",
        microstate_label="failed",
        phi_degrees=None,
        psi_degrees=None,
        reference_energy_kj_mol=float("nan"),
        perturbed_energy_kj_mol=float("nan"),
        delta_u_kj_mol=float("nan"),
        reference_relaxed_phi_degrees=None,
        reference_relaxed_psi_degrees=None,
        perturbed_relaxed_phi_degrees=None,
        perturbed_relaxed_psi_degrees=None,
        metadata={"status": "failed"},
    )

    results = aggregator.aggregate(
        [
            _frame_result(0, "a", 0.0),
            failed,
            _frame_result(1, "b", 10.0),
        ],
        atlas,
    )

    result_by_id = {result.state_id: result for result in results}
    assert result_by_id["a"].reference_probability == 0.5
    assert result_by_id["b"].reference_probability == 0.5
    assert not math.isnan(float(result_by_id["a"].reweight_factor))


def test_reweighting_uses_loaded_reference_probabilities() -> None:
    atlas = MicrostateAtlas(
        definitions=(
            MicrostateDefinition(
                "a",
                "a",
                (-180.0, 0.0),
                (-180.0, 180.0),
                reference_probability=0.25,
            ),
            MicrostateDefinition(
                "b",
                "b",
                (0.0, 180.0),
                (-180.0, 180.0),
                reference_probability=0.75,
            ),
        ),
        use_loaded_reference_probabilities=True,
    )
    aggregator = ReweightingAggregator(temperature_kelvin=300.0)
    results = aggregator.aggregate(
        [_frame_result(0, "a", 0.0), _frame_result(1, "b", 0.0)],
        atlas,
    )

    result_by_id = {result.state_id: result for result in results}
    assert abs(result_by_id["a"].reference_probability - 0.25) < 1.0e-8
    assert abs(result_by_id["b"].reference_probability - 0.75) < 1.0e-8


def test_reweighting_supports_discrete_states_with_plot_coordinates() -> None:
    atlas = MicrostateAtlas(
        definitions=(
            MicrostateDefinition(
                state_id="0",
                label="cluster_0",
                reference_probability=0.6,
                plot_x=-1.0,
                plot_y=0.5,
                state_level="cluster",
            ),
            MicrostateDefinition(
                state_id="1",
                label="cluster_1",
                reference_probability=0.4,
                plot_x=1.0,
                plot_y=-0.5,
                state_level="cluster",
            ),
        ),
        use_loaded_reference_probabilities=True,
    )
    aggregator = ReweightingAggregator(temperature_kelvin=300.0)
    results = aggregator.aggregate(
        [_frame_result(0, "0", 0.0), _frame_result(1, "1", 2.0)],
        atlas,
    )

    result_by_id = {result.state_id: result for result in results}
    assert result_by_id["0"].plot_x == -1.0
    assert result_by_id["1"].plot_y == -0.5
    assert result_by_id["0"].state_level == "cluster"


def test_load_microstate_definitions_supports_generalized_cv_ranges(
    tmp_path: Path,
) -> None:
    state_file = tmp_path / "states.json"
    state_file.write_text(
        json.dumps(
            {
                "states": [
                    {
                        "state_id": "helical",
                        "label": "helical",
                        "cv_ranges": {
                            "phi": [-120.0, -30.0],
                            "omega": [150.0, 180.0],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    definitions = load_microstate_definitions(str(state_file))

    assert len(definitions) == 1
    assert definitions[0].cv_ranges == {
        "phi": (-120.0, -30.0),
        "omega": (150.0, 180.0),
    }
    assignment = MicrostateAtlas(definitions=tuple(definitions)).assign(
        {"phi": -60.0, "omega": 170.0}
    )
    assert assignment.state_id == "helical"
    assert assignment.phi_degrees == -60.0
    assert assignment.measurements["omega"] == 170.0
