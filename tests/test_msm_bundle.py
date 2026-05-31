from __future__ import annotations

from pathlib import Path

import numpy as np

from felp.core import InputConfig, MicrostateConfig, MicrostateDefinition
from felp.io.trajectory import TrajectoryLoader
from felp.msm import MSMBundleManifest, load_bundle, save_bundle
from felp.reweighting import MicrostateAtlas


def _write_bundle(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    topology = tmp_path / "topology.pdb"
    trajectory_a = tmp_path / "traj_a.xtc"
    trajectory_b = tmp_path / "traj_b.xtc"
    topology.write_text("placeholder\n", encoding="utf-8")
    trajectory_a.write_text("placeholder\n", encoding="utf-8")
    trajectory_b.write_text("placeholder\n", encoding="utf-8")

    manifest = MSMBundleManifest(
        topology_file=str(topology),
        trajectory_groups=((str(trajectory_a),), (str(trajectory_b),)),
        stride=2,
        dt_traj="1 step",
        lag_time=5,
        feature_spec={
            "operations": [
                {
                    "kind": "distances",
                    "labels": ["ends", "turn"],
                    "pairs": [[0, 1], [2, 3]],
                }
            ]
        },
        reducer={"kind": "none"},
        clustering={"method": "kmeans"},
        model={"discrete_msm": {"active_state_fraction": 1.0}},
        state_definitions={
            "cluster": (
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
            "metastable": (
                MicrostateDefinition(
                    state_id="0",
                    label="meta_0",
                    reference_probability=0.75,
                    plot_x=-0.5,
                    plot_y=0.0,
                    state_level="metastable",
                ),
                MicrostateDefinition(
                    state_id="1",
                    label="meta_1",
                    reference_probability=0.25,
                    plot_x=0.5,
                    plot_y=0.0,
                    state_level="metastable",
                ),
            ),
        },
    )
    bundle_dir = tmp_path / "bundle"
    save_bundle(
        bundle_dir=str(bundle_dir),
        manifest=manifest,
        assignments={
            "cluster": (np.array([0, 1]), np.array([1])),
            "metastable": (np.array([0, 0]), np.array([1])),
        },
        frame_source_indices=(np.array([0, 2]), np.array([0])),
        embedding={
            "cluster_centers": np.array([[-1.0, 0.5], [1.0, -0.5]]),
            "feature_centers": np.array([[-1.0, 0.5], [1.0, -0.5]]),
            "metastable_feature_centers": np.array([[-0.5, 0.0], [0.5, 0.0]]),
        },
    )
    return bundle_dir, topology, trajectory_a, trajectory_b


def test_bundle_round_trip_and_lookup(tmp_path: Path) -> None:
    bundle_dir, _, _, _ = _write_bundle(tmp_path)

    bundle = load_bundle(str(bundle_dir))
    assert bundle.available_state_levels() == ("cluster", "metastable")
    assert bundle.lookup_state_id(0, 0, "cluster") == "0"
    assert bundle.lookup_state_id(0, 2, "cluster") == "1"
    assert bundle.lookup_state_id(1, 0, "metastable") == "1"
    assert bundle.summary()["state_counts"]["cluster"] == 2


def test_microstate_atlas_loads_bundle_definitions(tmp_path: Path) -> None:
    bundle_dir, _, _, _ = _write_bundle(tmp_path)
    atlas = MicrostateAtlas.from_config(
        MicrostateConfig(
            source="msm_bundle",
            msm_bundle_dir=str(bundle_dir),
            state_level="metastable",
        )
    )

    assert len(atlas.definitions) == 2
    assert atlas.definitions[0].state_level == "metastable"
    assert atlas.definitions[0].reference_probability == 0.75
    assert atlas.msm_assignment_index is not None


def test_microstate_atlas_assigns_msm_bundle_by_nearest_feature_centroid(
    tmp_path: Path,
) -> None:
    bundle_dir, _, _, _ = _write_bundle(tmp_path)
    atlas = MicrostateAtlas.from_config(
        MicrostateConfig(
            source="msm_bundle",
            msm_bundle_dir=str(bundle_dir),
            state_level="cluster",
        )
    )

    near_zero = atlas.assign({"ends": -10.0, "turn": 5.0})
    near_one = atlas.assign({"ends": 10.0, "turn": -5.0})

    assert near_zero.state_id == "0"
    assert near_one.state_id == "1"
    assert atlas.definitions[0].cv_ranges["ends"] == (-10.0, -10.0)
    assert atlas.definitions[0].cv_ranges["turn"] == (5.0, 5.0)


def test_microstate_atlas_assigns_msm_metastable_level(tmp_path: Path) -> None:
    bundle_dir, _, _, _ = _write_bundle(tmp_path)
    atlas = MicrostateAtlas.from_config(
        MicrostateConfig(
            source="msm_bundle",
            msm_bundle_dir=str(bundle_dir),
            state_level="metastable",
        )
    )

    assignment = atlas.assign({"ends": -5.0, "turn": 0.0})
    assert assignment.state_id == "0"


def test_microstate_atlas_uses_cluster_centers_fallback_for_legacy_bundles(
    tmp_path: Path,
) -> None:
    bundle_dir, _, _, _ = _write_bundle(tmp_path)
    embedding_path = bundle_dir / "embedding.npz"
    embedding = dict(np.load(embedding_path))
    del embedding["feature_centers"]
    del embedding["metastable_feature_centers"]
    np.savez(embedding_path, **embedding)

    atlas = MicrostateAtlas.from_config(
        MicrostateConfig(
            source="msm_bundle",
            msm_bundle_dir=str(bundle_dir),
            state_level="cluster",
        )
    )

    assignment = atlas.assign({"ends": -10.0, "turn": 5.0})
    assert assignment.state_id == "0"
    assert atlas.definitions[0].cv_ranges["ends"] == (-10.0, -10.0)
    assert atlas.definitions[0].cv_ranges["turn"] == (5.0, 5.0)


def test_trajectory_loader_attaches_precomputed_states(tmp_path: Path, monkeypatch) -> None:
    bundle_dir, topology, trajectory_a, trajectory_b = _write_bundle(tmp_path)
    bundle = load_bundle(str(bundle_dir))

    class FakeFrame:
        def __init__(self, xyz):
            self.xyz = xyz

    class FakeMD:
        @staticmethod
        def load(path: str, top: str, stride: int):
            assert top == str(topology)
            assert stride == 2
            if path == str(trajectory_a):
                return [
                    FakeFrame(np.array([[[0.0, 0.0, 0.0]]])),
                    FakeFrame(np.array([[[1.0, 0.0, 0.0]]])),
                ]
            if path == str(trajectory_b):
                return [FakeFrame(np.array([[[2.0, 0.0, 0.0]]]))]
            raise AssertionError(f"Unexpected trajectory path: {path}")

    monkeypatch.setattr("felp.io.trajectory._import_mdtraj", lambda: FakeMD)
    jobs = TrajectoryLoader(
        InputConfig(
            topology_file=str(topology),
            trajectory_files=((str(trajectory_a),), (str(trajectory_b),)),
            stride=2,
        ),
        bundle=bundle,
        state_level="cluster",
    ).load_jobs()

    assert [job.precomputed_state_id for job in jobs] == ["0", "1", "1"]
    assert [job.trajectory_id for job in jobs] == [0, 0, 1]
    assert [job.source_frame_index for job in jobs] == [0, 2, 0]


def test_trajectory_loader_streams_grouped_segments_with_global_stride(
    tmp_path: Path,
    monkeypatch,
) -> None:
    topology = tmp_path / "topology.pdb"
    trajectory_a = tmp_path / "traj_a.xtc"
    trajectory_b = tmp_path / "traj_b.xtc"
    topology.write_text("placeholder\n", encoding="utf-8")
    trajectory_a.write_text("placeholder\n", encoding="utf-8")
    trajectory_b.write_text("placeholder\n", encoding="utf-8")

    class FakeChunk:
        def __init__(self, values: list[float]):
            self.xyz = np.asarray(
                [[[value, 0.0, 0.0]] for value in values],
                dtype=float,
            )

    class FakeMD:
        calls: list[tuple[str, int, int]] = []

        @staticmethod
        def iterload(path: str, chunk: int, top: str, stride: int):
            assert top == str(topology)
            FakeMD.calls.append((path, chunk, stride))
            values = {
                str(trajectory_a): [0.0, 1.0, 2.0],
                str(trajectory_b): [3.0, 4.0, 5.0],
            }[path]
            selected = values[::stride]
            for start in range(0, len(selected), chunk):
                yield FakeChunk(selected[start : start + chunk])

    monkeypatch.setattr("felp.io.trajectory._import_mdtraj", lambda: FakeMD)

    jobs = TrajectoryLoader(
        InputConfig(
            topology_file=str(topology),
            trajectory_files=((str(trajectory_a), str(trajectory_b)),),
            stride=2,
        ),
        load_chunk_size=2,
    ).load_jobs()

    assert FakeMD.calls == [
        (str(trajectory_a), 2, 1),
        (str(trajectory_b), 2, 1),
    ]
    assert [job.trajectory_id for job in jobs] == [0, 0, 0]
    assert [job.source_frame_index for job in jobs] == [0, 2, 4]
    assert [job.coordinates_angstrom[0][0] for job in jobs] == [0.0, 20.0, 40.0]
