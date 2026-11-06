"""Tests for CV-grid discretization and ND neighbor topology."""

from __future__ import annotations

import numpy as np
import pytest

from felp.core import MicrostateDefinition
from felp.msm.config import MSMFitConfig
from felp.reweighting import MicrostateAtlas
from felp.reweighting.grid_microstates import (
    build_grid_discretization,
    grid_cluster_trajectories,
)
from felp.reweighting.topology import build_grid_topology, grid_neighbor_offsets


def _definitions_from_bin_grid(
    cv_names: tuple[str, ...],
    shape: tuple[int, ...],
) -> tuple[MicrostateDefinition, ...]:
    definitions: list[MicrostateDefinition] = []
    index_ranges = [range(count) for count in shape]
    import itertools

    for multi_index in itertools.product(*index_ranges):
        bin_indices = {name: int(multi_index[axis]) for axis, name in enumerate(cv_names)}
        cv_ranges = {
            name: (float(multi_index[axis]), float(multi_index[axis]) + 1.0)
            for axis, name in enumerate(cv_names)
        }
        state_id = "_".join(
            f"{name}_{multi_index[axis]:02d}" for axis, name in enumerate(cv_names)
        )
        definitions.append(
            MicrostateDefinition(
                state_id=state_id,
                label=state_id,
                cv_ranges=cv_ranges,
                metadata={"bin_indices": bin_indices},
            )
        )
    return tuple(definitions)


def test_grid_neighbor_offsets_1d_cardinal_and_chebyshev3() -> None:
    cardinal = grid_neighbor_offsets(ndim=1, connectivity="cardinal", chebyshev_radius=0)
    assert cardinal == ((-1,), (1,))
    cheb3 = grid_neighbor_offsets(ndim=1, connectivity="chebyshev", chebyshev_radius=3)
    assert len(cheb3) == 6


def test_grid_neighbor_offsets_3d_cardinal() -> None:
    cardinal = grid_neighbor_offsets(ndim=3, connectivity="cardinal", chebyshev_radius=0)
    assert len(cardinal) == 6


def test_topology_1d_bin_indices() -> None:
    atlas = MicrostateAtlas.from_definitions(
        _definitions_from_bin_grid(("lambda",), (5,)),
        reference_probability_mode="raw_counts",
    )
    topology = build_grid_topology(atlas, grid_connectivity="cardinal")
    assert topology is not None
    assert topology.grid_shape == (5,)
    assert topology.cv_names == ("lambda",)
    center = atlas.definitions[2].state_id
    assert set(topology.neighbor_ids(center)) == {
        atlas.definitions[1].state_id,
        atlas.definitions[3].state_id,
    }


def test_topology_3d_chebyshev1() -> None:
    atlas = MicrostateAtlas.from_definitions(
        _definitions_from_bin_grid(("a", "b", "c"), (3, 3, 3)),
    )
    topology = build_grid_topology(atlas, grid_connectivity="chebyshev1")
    assert topology is not None
    assert topology.grid_shape == (3, 3, 3)
    center = atlas.definitions[13].state_id  # (1,1,1)
    assert len(topology.neighbor_ids(center)) == 26


def test_grid_cluster_trajectories_assigns_bins() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=(200, 2))
    trajectories = (values,)
    feature_spec = {
        "operations": [
            {"kind": "distances", "labels": ["ends", "turn"], "pairs": [[0, 1], [2, 3]]}
        ]
    }
    result = grid_cluster_trajectories(
        trajectories,
        feature_spec=feature_spec,
        grid_bins=(4, 4),
    )
    assert result.clustercenters.shape == (16, 2)
    assert result.dtrajs[0].shape == (200,)
    assert set(result.dtrajs[0]).issubset(set(range(16)))


def test_msm_fit_config_grid_requires_reducer_none(tmp_path) -> None:
    topo = tmp_path / "top.pdb"
    traj = tmp_path / "traj.xtc"
    spec = tmp_path / "spec.json"
    topo.write_text("x", encoding="utf-8")
    traj.write_text("x", encoding="utf-8")
    spec.write_text('{"operations": []}', encoding="utf-8")
    with pytest.raises(Exception, match="reducer=none"):
        MSMFitConfig(
            topology_file=str(topo),
            trajectory_files=((str(traj),),),
            output_dir=str(tmp_path / "out"),
            feature_spec_file=str(spec),
            clustering_method="grid",
            reducer="tica",
        ).validate()


def test_build_grid_discretization_state_ids_match_auto_format() -> None:
    rows = np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    disc = build_grid_discretization(rows, ("ends", "turn"), bin_counts=(2, 2))
    assert disc.state_ids[0] == "ends_00_turn_00"
    assert disc.state_count == 4
