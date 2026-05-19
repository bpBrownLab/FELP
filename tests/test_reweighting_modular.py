"""Tests for modular reweighting backends."""

from __future__ import annotations

import json
import os
import statistics
import tempfile

import pytest

from felp.core import (
    FrameResult,
    LaplacianConfig,
    MicrostateDefinition,
    NeighborPoolConfig,
    ReweightingConfig,
    ReweightingQCConfig,
)
from felp.io.results import ResultsWriter
from felp.reweighting import MicrostateAtlas, ReweightingAggregator, ReweightingOrchestrator
from felp.reweighting.topology import build_grid_topology
from felp.reweighting.engines import get_reweighting_engine
from felp.reweighting.prep import build_reweighting_prep


def _frame_result(index: int, state_id: str, delta_u: float) -> FrameResult:
    return FrameResult(
        frame_index=index,
        stride_number=index,
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
        assignment_source="test",
        state_level="microstate",
        metadata={"status": "completed"},
    )


def _grid_atlas() -> MicrostateAtlas:
    definitions = []
    for x_index, x_center in enumerate((-120.0, 0.0, 120.0)):
        for y_index, y_center in enumerate((-120.0, 0.0, 120.0)):
            state_id = f"phi_{x_index:02d}_psi_{y_index:02d}"
            definitions.append(
                MicrostateDefinition(
                    state_id=state_id,
                    label=state_id,
                    cv_ranges={
                        "phi": (x_center - 60.0, x_center + 60.0),
                        "psi": (y_center - 60.0, y_center + 60.0),
                    },
                    plot_x=x_center,
                    plot_y=y_center,
                    metadata={"bin_indices": {"phi": x_index, "psi": y_index}},
                )
            )
    return MicrostateAtlas.from_definitions(
        definitions,
        reference_probability_mode="raw_counts",
    )


def test_topology_from_bin_indices() -> None:
    atlas = _grid_atlas()
    topology = build_grid_topology(atlas)
    assert topology is not None
    assert topology.source == "bin_indices"
    assert topology.grid_shape == (3, 3)
    center_id = "phi_01_psi_01"
    neighbors = set(topology.neighbor_ids(center_id))
    assert neighbors == {
        "phi_00_psi_01",
        "phi_02_psi_01",
        "phi_01_psi_00",
        "phi_01_psi_02",
    }


def test_independent_matches_legacy_aggregator() -> None:
    atlas = MicrostateAtlas(
        definitions=(
            MicrostateDefinition("a", "a", (-180.0, 0.0), (-180.0, 180.0)),
            MicrostateDefinition("b", "b", (0.0, 180.0), (-180.0, 180.0)),
        ),
        reference_probability_mode="raw_counts",
    )
    frames = [
        *(_frame_result(index, "a", 0.0) for index in range(40)),
        *(_frame_result(index, "b", 5.0) for index in range(40, 49)),
    ]
    legacy = ReweightingAggregator(temperature_kelvin=300.0).aggregate(frames, atlas)
    engine = get_reweighting_engine("independent")
    prep = build_reweighting_prep(
        frames,
        atlas,
        beta=1.0 / (0.00831446261815324 * 300.0),
        temperature_kelvin=300.0,
        qc_config=ReweightingQCConfig(),
    )
    outcome = engine.aggregate(
        prep,
        config=ReweightingConfig(methods=("independent",)),
    )
    assert len(outcome.microstate_results) == len(legacy)
    for new, old in zip(
        sorted(outcome.microstate_results, key=lambda item: item.state_id),
        sorted(legacy, key=lambda item: item.state_id),
        strict=True,
    ):
        assert new.target_probability == pytest.approx(old.target_probability)
        assert new.reweight_factor == pytest.approx(old.reweight_factor)


def test_neighbor_pool_reduces_variance_on_synthetic_grid() -> None:
    atlas = _grid_atlas()
    frames = []
    index = 0
    for definition in atlas.definitions:
        delta_u = 0.0 if definition.state_id.endswith("_01") else 20.0
        for _ in range(5):
            frames.append(_frame_result(index, definition.state_id, delta_u))
            index += 1
    prep = build_reweighting_prep(
        frames,
        atlas,
        beta=1.0 / (0.00831446261815324 * 300.0),
        temperature_kelvin=300.0,
        qc_config=ReweightingQCConfig(),
    )
    independent = get_reweighting_engine("independent").aggregate(
        prep,
        config=ReweightingConfig(methods=("independent",)),
    )
    pooled = get_reweighting_engine("neighbor_pool").aggregate(
        prep,
        config=ReweightingConfig(
            methods=("neighbor_pool",),
            neighbor_pool=NeighborPoolConfig(tau=50.0),
        ),
    )
    independent_factors = [
        result.reweight_factor for result in independent.microstate_results
    ]
    pooled_factors = [result.reweight_factor for result in pooled.microstate_results]
    assert statistics.pstdev(pooled_factors) < statistics.pstdev(independent_factors)


def test_laplacian_positive_lambda_smooths_on_grid() -> None:
    atlas = _grid_atlas()
    frames = []
    index = 0
    for definition in atlas.definitions:
        # Strong checkerboard noise in local factors via delta_u
        parity = (int(definition.metadata["bin_indices"]["phi"]) +
                  int(definition.metadata["bin_indices"]["psi"])) % 2
        delta_u = 0.0 if parity == 0 else 15.0
        for _ in range(8):
            frames.append(_frame_result(index, definition.state_id, delta_u))
            index += 1
    prep = build_reweighting_prep(
        frames,
        atlas,
        beta=1.0 / (0.00831446261815324 * 300.0),
        temperature_kelvin=300.0,
        qc_config=ReweightingQCConfig(),
    )
    independent = get_reweighting_engine("independent").aggregate(
        prep, config=ReweightingConfig(methods=("independent",))
    )
    laplacian = get_reweighting_engine("laplacian").aggregate(
        prep,
        config=ReweightingConfig(
            methods=("laplacian",),
            laplacian=LaplacianConfig(
                lambda_kj_mol2=2.0,
                lambda_scale_by_kbt=True,
            ),
        ),
    )
    ind_factors = [r.reweight_factor for r in independent.microstate_results]
    lap_factors = [r.reweight_factor for r in laplacian.microstate_results]
    assert statistics.pstdev(lap_factors) < statistics.pstdev(ind_factors)
    assert laplacian.shared_metadata["laplacian"]["solver"] != "analytic_independent_limit"


def test_neighbor_pool_gcv_tau_selection() -> None:
    atlas = _grid_atlas()
    frames = []
    index = 0
    for definition in atlas.definitions:
        delta_u = 5.0 if definition.state_id.endswith("_01") else 0.0
        for _ in range(5):
            frames.append(_frame_result(index, definition.state_id, delta_u))
            index += 1
    prep = build_reweighting_prep(
        frames,
        atlas,
        beta=1.0 / (0.00831446261815324 * 300.0),
        temperature_kelvin=300.0,
        qc_config=ReweightingQCConfig(),
    )
    outcome = get_reweighting_engine("neighbor_pool").aggregate(
        prep,
        config=ReweightingConfig(
            methods=("neighbor_pool",),
            neighbor_pool=NeighborPoolConfig(tau=None, tau_mode="gcv"),
        ),
    )
    tau_meta = outcome.shared_metadata["neighbor_pool"]
    assert tau_meta["tau_mode"] == "gcv"
    assert float(tau_meta["tau"]) > 0.0
    assert "gcv_best_tau" in tau_meta


def test_laplacian_lambda_zero_near_independent() -> None:
    atlas = MicrostateAtlas(
        definitions=(
            MicrostateDefinition("a", "a", (-180.0, 0.0), (-180.0, 180.0)),
            MicrostateDefinition("b", "b", (0.0, 180.0), (-180.0, 180.0)),
        ),
        reference_probability_mode="raw_counts",
    )
    frames = [
        *(_frame_result(index, "a", 0.0) for index in range(20)),
        *(_frame_result(index, "b", 2.0) for index in range(20, 40)),
    ]
    prep = build_reweighting_prep(
        frames,
        atlas,
        beta=1.0 / (0.00831446261815324 * 300.0),
        temperature_kelvin=300.0,
        qc_config=ReweightingQCConfig(),
    )
    independent = get_reweighting_engine("independent").aggregate(
        prep, config=ReweightingConfig(methods=("independent",))
    )
    laplacian = get_reweighting_engine("laplacian").aggregate(
        prep,
        config=ReweightingConfig(
            methods=("laplacian",),
            laplacian=LaplacianConfig(lambda_kj_mol2=0.0),
        ),
    )
    for ind, lap in zip(
        sorted(independent.microstate_results, key=lambda item: item.state_id),
        sorted(laplacian.microstate_results, key=lambda item: item.state_id),
        strict=True,
    ):
        assert lap.target_probability == pytest.approx(ind.target_probability, rel=0.05)


def test_frame_method_writes_weights_and_normalizes() -> None:
    atlas = MicrostateAtlas(
        definitions=(
            MicrostateDefinition("a", "a", (-180.0, 0.0), (-180.0, 180.0)),
            MicrostateDefinition("b", "b", (0.0, 180.0), (-180.0, 180.0)),
        ),
        reference_probability_mode="raw_counts",
    )
    frames = [
        *(_frame_result(index, "a", 0.0) for index in range(20)),
        *(_frame_result(index, "b", 1.0) for index in range(20, 40)),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        orchestrator = ReweightingOrchestrator(
            temperature_kelvin=300.0,
            config=ReweightingConfig(methods=("frame",)),
            qc_config=ReweightingQCConfig(),
        )
        writer = ResultsWriter(
            __import__("felp.core", fromlist=["OutputConfig"]).OutputConfig(
                output_dir=tmpdir
            )
        )
        outcome = orchestrator.run_single(frames, atlas)
        writer.write_microstate_results(outcome.microstate_results)
        writer.write_frame_weights(outcome.auxiliary["frame_weights"])
        total_target = sum(result.target_probability for result in outcome.microstate_results)
        assert total_target == pytest.approx(1.0)
        weights_path = os.path.join(tmpdir, "frame_weights.csv")
        assert os.path.isfile(weights_path)


def test_orchestrator_multi_method_writes_comparison() -> None:
    atlas = MicrostateAtlas(
        definitions=(
            MicrostateDefinition("a", "a", (-180.0, 0.0), (-180.0, 180.0)),
            MicrostateDefinition("b", "b", (0.0, 180.0), (-180.0, 180.0)),
        ),
        reference_probability_mode="raw_counts",
    )
    frames = [
        *(_frame_result(index, "a", 0.0) for index in range(20)),
        *(_frame_result(index, "b", 1.0) for index in range(20, 40)),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = ResultsWriter(
            __import__("felp.core", fromlist=["OutputConfig"]).OutputConfig(
                output_dir=tmpdir
            )
        )
        ReweightingOrchestrator(
            temperature_kelvin=300.0,
            config=ReweightingConfig(methods=("independent", "neighbor_pool")),
        ).run(
            frames,
            atlas,
            output_dir=tmpdir,
            writer=writer,
        )
        comparison_path = os.path.join(tmpdir, "reweighting_comparison.json")
        assert os.path.isfile(comparison_path)
        payload = json.loads(open(comparison_path, encoding="utf-8").read())
        assert set(payload["methods"]) == {"independent", "neighbor_pool"}
        assert os.path.isdir(os.path.join(tmpdir, "reweighting_independent"))
        assert os.path.isdir(os.path.join(tmpdir, "reweighting_neighbor_pool"))
