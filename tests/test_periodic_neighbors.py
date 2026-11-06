"""Periodic topology and estimator reconstruction across the public workflow."""

from dataclasses import asdict, replace

import numpy as np
import pytest

from felp.cli.parser import build_parser, build_reweighting_config
from felp.core import (
    ConfigurationError, MicrostateDefinition, NeighborPoolConfig,
    ReweightingConfig, ReweightingQCConfig,
)
from felp.reweighting import MicrostateAtlas, ReweightingOrchestrator
from felp.reweighting.topology import build_grid_topology
from tests.test_reweighting_modular import _frame_result, _grid_atlas


@pytest.mark.parametrize("use_indices", [True, False])
def test_cardinal_wraps_only_requested_axis(use_indices):
    atlas = _grid_atlas()
    if not use_indices:
        atlas = MicrostateAtlas.from_definitions(
            [replace(d, metadata={}) for d in reversed(atlas.definitions)]
        )
    ordinary = build_grid_topology(atlas)
    wrapped = build_grid_topology(atlas, periodic_cv_names=("phi",))
    assert ordinary.connectivity == wrapped.connectivity == "cardinal"
    assert set(ordinary.neighbor_ids("phi_00_psi_00")) == {
        "phi_01_psi_00", "phi_00_psi_01",
    }
    assert set(wrapped.neighbor_ids("phi_00_psi_00")) == {
        "phi_01_psi_00", "phi_02_psi_00", "phi_00_psi_01",
    }
    assert wrapped.periodic_cv_names == ("phi",)


def test_wide_periodic_stencil_has_unique_neighbors_and_no_self_edges():
    topology = build_grid_topology(
        _grid_atlas(), grid_connectivity="chebyshev3",
        periodic_cv_names=("phi", "psi"),
    )
    for state in topology.neighbors:
        neighbors = topology.neighbor_ids(state)
        assert len(neighbors) == len(set(neighbors)) == 8
        assert state not in neighbors


def test_single_bin_periodic_axis_has_no_self_edge():
    atlas = MicrostateAtlas.from_definitions([
        MicrostateDefinition("one", "one", cv_ranges={"aloop": (-180, 180)})
    ])
    topology = build_grid_topology(atlas, periodic_cv_names=("aloop",))
    assert topology.neighbor_ids("one") == ()


def test_periodic_axes_are_validated_instead_of_silently_ignored():
    with pytest.raises(ConfigurationError, match="not present"):
        build_grid_topology(_grid_atlas(), periodic_cv_names=("aloop",))
    atlas = MicrostateAtlas.from_definitions([
        MicrostateDefinition("a", "a", plot_x=0, plot_y=0),
        MicrostateDefinition("b", "b", plot_x=1, plot_y=1),
    ])
    with pytest.raises(ConfigurationError, match="regular CV grid"):
        build_grid_topology(atlas, periodic_cv_names=("aloop",))


@pytest.mark.parametrize("command", ["run", "summarize"])
def test_cli_and_serialization_keep_cardinal_with_explicit_periodicity(command):
    parser = build_parser()
    if command == "run":
        required = ["--topology-file", "top.pdb", "--trajectory", "traj.nc",
                    "--target-residue-number", "2", "--target-residue-name", "ALA"]
    else:
        required = ["--raw-frame-results", "frames.csv"]
    namespace = parser.parse_args([
        command, *required, "--output-dir", "out",
        "--neighbor-pool-periodic-cvs", "phi", "psi", "phi",
    ])
    pool = build_reweighting_config(namespace).neighbor_pool
    assert pool.grid_connectivity == "cardinal"
    assert pool.periodic_cv_names == ("phi", "psi")
    assert NeighborPoolConfig.from_dict(asdict(pool)) == pool
    assert NeighborPoolConfig.from_dict({}).grid_connectivity == "cardinal"


def test_periodicity_changes_boundary_estimate_and_survives_config_roundtrip():
    atlas = _grid_atlas()
    frames = [
        _frame_result(i * 30 + j, definition.state_id, float(i))
        for i, definition in enumerate(atlas.definitions) for j in range(30)
    ]
    qc = ReweightingQCConfig(policy="flag", weight_treatment="gpd-psis")

    def run(pool):
        return ReweightingOrchestrator(
            310, config=ReweightingConfig(methods=("neighbor_pool",), neighbor_pool=pool),
            qc_config=qc,
        ).run(frames, atlas)["neighbor_pool"]

    ordinary = run(NeighborPoolConfig(tau=100))
    wrapped = run(NeighborPoolConfig(tau=100, periodic_cv_names=("phi",)))
    summary = wrapped.shared_metadata
    assert summary["neighbor_pool"]["periodic_cv_names"] == ["phi"]
    assert summary["neighbor_pool"]["grid_connectivity"] == "cardinal"
    config = ReweightingConfig.from_dict(asdict(ReweightingConfig(
        methods=("neighbor_pool",), neighbor_pool=NeighborPoolConfig(tau=100, periodic_cv_names=("phi",)))))
    assert config.neighbor_pool.periodic_cv_names == ("phi",)
    assert config.neighbor_pool.tau == 100
    reconstructed = run(config.neighbor_pool)
    probabilities = lambda result: [x.target_probability for x in result.microstate_results]
    assert not np.allclose(probabilities(ordinary), probabilities(wrapped))
    np.testing.assert_allclose(probabilities(reconstructed), probabilities(wrapped))
    assert sum(probabilities(wrapped)) == pytest.approx(1)
