from __future__ import annotations

from felp.core import FrameResult, MicrostateDefinition
from felp.reweighting import MicrostateAtlas, ReweightingAggregator


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
