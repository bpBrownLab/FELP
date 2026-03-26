from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from felp.core import MicrostateDefinition
from felp.msm import MSMBundleManifest, save_bundle
from felp.plotting import FELPPlotConfig, MSMPlotConfig, generate_felp_plots, generate_msm_plots


pytest.importorskip("matplotlib")


def _write_bundle(tmp_path: Path) -> Path:
    topology = tmp_path / "topology.pdb"
    trajectory = tmp_path / "traj.xtc"
    topology.write_text("placeholder\n", encoding="utf-8")
    trajectory.write_text("placeholder\n", encoding="utf-8")

    manifest = MSMBundleManifest(
        topology_file=str(topology),
        trajectory_groups=((str(trajectory),),),
        stride=1,
        dt_traj="1 step",
        lag_time=5,
        feature_spec={"operations": []},
        reducer={"kind": "tica"},
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
            )
        },
    )
    bundle_dir = tmp_path / "bundle"
    save_bundle(
        bundle_dir=str(bundle_dir),
        manifest=manifest,
        assignments={"cluster": (np.array([0, 1, 0, 1]),)},
        frame_source_indices=(np.array([0, 1, 2, 3]),),
        embedding={
            "cluster_centers": np.array([[-1.0, 0.5], [1.0, -0.5]]),
            "reduced_trajectory_0": np.array(
                [[-1.0, 0.5], [0.8, -0.4], [-0.9, 0.6], [1.1, -0.6]]
            ),
        },
    )
    return bundle_dir


def test_generate_msm_plots(tmp_path: Path) -> None:
    bundle_dir = _write_bundle(tmp_path)

    summary = generate_msm_plots(
        MSMPlotConfig(
            bundle_dir=str(bundle_dir),
            output_dir=str(tmp_path / "plots"),
            state_level="cluster",
            bins=20,
            max_points=100,
        )
    )

    assert summary["state_count"] == 2
    assert (tmp_path / "plots" / "state_probabilities.png").is_file()
    assert (tmp_path / "plots" / "reference_free_energy_landscape.png").is_file()
    assert (tmp_path / "plots" / "plot_summary.json").is_file()


def test_generate_felp_plots(tmp_path: Path) -> None:
    summary_csv = tmp_path / "microstate_summary.csv"
    frame_csv = tmp_path / "frame_results.csv"

    with open(summary_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "label",
                "reference_probability",
                "target_probability",
                "free_energy_kj_mol",
                "plot_x",
                "plot_y",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "label": "cluster_0",
                "reference_probability": 0.6,
                "target_probability": 0.45,
                "free_energy_kj_mol": 0.2,
                "plot_x": -1.0,
                "plot_y": 0.5,
            }
        )
        writer.writerow(
            {
                "label": "cluster_1",
                "reference_probability": 0.4,
                "target_probability": 0.55,
                "free_energy_kj_mol": 0.0,
                "plot_x": 1.0,
                "plot_y": -0.5,
            }
        )

    with open(frame_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["delta_u_kj_mol"])
        writer.writeheader()
        writer.writerow({"delta_u_kj_mol": -0.5})
        writer.writerow({"delta_u_kj_mol": 0.1})
        writer.writerow({"delta_u_kj_mol": 0.8})

    summary = generate_felp_plots(
        FELPPlotConfig(
            microstate_summary_csv=str(summary_csv),
            raw_frame_results_csv=str(frame_csv),
            output_dir=str(tmp_path / "felp_plots"),
            top_states=10,
        )
    )

    assert "kl_reference_to_target" in summary
    assert (tmp_path / "felp_plots" / "occupancy_shifts.png").is_file()
    assert (tmp_path / "felp_plots" / "delta_u_distribution.png").is_file()
    assert (tmp_path / "felp_plots" / "plot_summary.json").is_file()
