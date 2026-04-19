from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from felp.core import FrameResult, MicrostateDefinition, OutputConfig
from felp.io import ResultsWriter
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
                "state_id",
                "label",
                "frame_count",
                "reference_probability",
                "target_probability",
                "free_energy_kj_mol",
                "plot_x",
                "plot_y",
                "phi_center",
                "psi_center",
                "metadata",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "state_id": "cluster_0",
                "label": "cluster_0",
                "frame_count": 2,
                "reference_probability": 0.6,
                "target_probability": 0.45,
                "free_energy_kj_mol": 0.2,
                "plot_x": -1.0,
                "plot_y": 0.5,
                "phi_center": -60.0,
                "psi_center": -40.0,
                "metadata": (
                    '{"prior_probability_mode":"loaded",'
                    '"reference_probability_source":"loaded"}'
                ),
            }
        )
        writer.writerow(
            {
                "state_id": "cluster_1",
                "label": "cluster_1",
                "frame_count": 1,
                "reference_probability": 0.4,
                "target_probability": 0.55,
                "free_energy_kj_mol": 0.0,
                "plot_x": 1.0,
                "plot_y": -0.5,
                "phi_center": 50.0,
                "psi_center": 40.0,
                "metadata": (
                    '{"prior_probability_mode":"loaded",'
                    '"reference_probability_source":"loaded"}'
                ),
            }
        )

    writer = ResultsWriter(OutputConfig(output_dir=str(tmp_path / "frame_out")))
    writer.prepare_output_dir()
    writer.write_frame_results(
        [
            FrameResult(
                frame_index=0,
                stride_number=0,
                microstate_id="cluster_0",
                microstate_label="cluster_0",
                phi_degrees=-58.0,
                psi_degrees=-38.0,
                reference_energy_kj_mol=0.0,
                perturbed_energy_kj_mol=-0.5,
                delta_u_kj_mol=-0.5,
                reference_relaxed_phi_degrees=-58.0,
                reference_relaxed_psi_degrees=-38.0,
                perturbed_relaxed_phi_degrees=-58.0,
                perturbed_relaxed_psi_degrees=-38.0,
            ),
            FrameResult(
                frame_index=1,
                stride_number=1,
                microstate_id="cluster_0",
                microstate_label="cluster_0",
                phi_degrees=-62.0,
                psi_degrees=-42.0,
                reference_energy_kj_mol=0.0,
                perturbed_energy_kj_mol=0.1,
                delta_u_kj_mol=0.1,
                reference_relaxed_phi_degrees=-62.0,
                reference_relaxed_psi_degrees=-42.0,
                perturbed_relaxed_phi_degrees=-62.0,
                perturbed_relaxed_psi_degrees=-42.0,
            ),
            FrameResult(
                frame_index=2,
                stride_number=2,
                microstate_id="cluster_1",
                microstate_label="cluster_1",
                phi_degrees=48.0,
                psi_degrees=38.0,
                reference_energy_kj_mol=0.0,
                perturbed_energy_kj_mol=0.8,
                delta_u_kj_mol=0.8,
                reference_relaxed_phi_degrees=48.0,
                reference_relaxed_psi_degrees=38.0,
                perturbed_relaxed_phi_degrees=48.0,
                perturbed_relaxed_psi_degrees=38.0,
            ),
        ]
    )
    frame_csv.write_text(
        Path(writer.raw_frame_results_path).read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    summary = generate_felp_plots(
        FELPPlotConfig(
            microstate_summary_csv=str(summary_csv),
            raw_frame_results_csv=str(frame_csv),
            output_dir=str(tmp_path / "felp_plots"),
            top_states=10,
        )
    )

    assert "kl_reference_to_target" in summary
    assert summary["reference_probability_source"] == "loaded"
    assert summary["phi_psi_frame_projection_available"] is True
    assert (tmp_path / "felp_plots" / "occupancy_shifts.png").is_file()
    assert (tmp_path / "felp_plots" / "delta_u_distribution.png").is_file()
    assert (tmp_path / "felp_plots" / "reference_phi_psi_free_energy_landscape.png").is_file()
    assert (tmp_path / "felp_plots" / "reweighted_phi_psi_free_energy_landscape.png").is_file()
    assert (tmp_path / "felp_plots" / "discrete_microstate_free_energy_ramachandran.png").is_file()
    assert (tmp_path / "felp_plots" / "plot_summary.json").is_file()


def test_generate_felp_plots_fel_energy_cap(tmp_path: Path) -> None:
    """Capped colormaps should still produce all expected FEL outputs."""

    summary_csv = tmp_path / "microstate_summary.csv"
    frame_csv = tmp_path / "frame_results.csv"

    with open(summary_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "state_id",
                "label",
                "frame_count",
                "reference_probability",
                "target_probability",
                "free_energy_kj_mol",
                "plot_x",
                "plot_y",
                "phi_center",
                "psi_center",
                "metadata",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "state_id": "cluster_0",
                "label": "cluster_0",
                "frame_count": 2,
                "reference_probability": 0.6,
                "target_probability": 0.45,
                "free_energy_kj_mol": 50.0,
                "plot_x": -1.0,
                "plot_y": 0.5,
                "phi_center": -60.0,
                "psi_center": -40.0,
                "metadata": '{"reference_probability_source":"loaded"}',
            }
        )
        writer.writerow(
            {
                "state_id": "cluster_1",
                "label": "cluster_1",
                "frame_count": 1,
                "reference_probability": 0.4,
                "target_probability": 0.55,
                "free_energy_kj_mol": 80.0,
                "plot_x": 1.0,
                "plot_y": -0.5,
                "phi_center": 50.0,
                "psi_center": 40.0,
                "metadata": '{"reference_probability_source":"loaded"}',
            }
        )

    writer = ResultsWriter(OutputConfig(output_dir=str(tmp_path / "frame_out")))
    writer.prepare_output_dir()
    writer.write_frame_results(
        [
            FrameResult(
                frame_index=0,
                stride_number=0,
                microstate_id="cluster_0",
                microstate_label="cluster_0",
                phi_degrees=-58.0,
                psi_degrees=-38.0,
                reference_energy_kj_mol=0.0,
                perturbed_energy_kj_mol=-0.5,
                delta_u_kj_mol=-0.5,
                reference_relaxed_phi_degrees=-58.0,
                reference_relaxed_psi_degrees=-38.0,
                perturbed_relaxed_phi_degrees=-58.0,
                perturbed_relaxed_psi_degrees=-38.0,
            ),
            FrameResult(
                frame_index=1,
                stride_number=1,
                microstate_id="cluster_1",
                microstate_label="cluster_1",
                phi_degrees=48.0,
                psi_degrees=38.0,
                reference_energy_kj_mol=0.0,
                perturbed_energy_kj_mol=0.8,
                delta_u_kj_mol=0.8,
                reference_relaxed_phi_degrees=48.0,
                reference_relaxed_psi_degrees=38.0,
                perturbed_relaxed_phi_degrees=48.0,
                perturbed_relaxed_psi_degrees=38.0,
            ),
        ]
    )
    frame_csv.write_text(
        Path(writer.raw_frame_results_path).read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    summary = generate_felp_plots(
        FELPPlotConfig(
            microstate_summary_csv=str(summary_csv),
            raw_frame_results_csv=str(frame_csv),
            output_dir=str(tmp_path / "felp_plots"),
            fel_max_energy_kj_mol=12.0,
        )
    )

    assert summary["fel_max_energy_kj_mol"] == 12.0
    assert (tmp_path / "felp_plots" / "reference_phi_psi_free_energy_landscape.png").is_file()
    assert (tmp_path / "felp_plots" / "delta_free_energy_projection.png").is_file()


def test_generate_felp_plots_plot_energy_unit_kcal_mol(tmp_path: Path) -> None:
    summary_csv = tmp_path / "microstate_summary.csv"

    with open(summary_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "state_id",
                "label",
                "frame_count",
                "reference_probability",
                "target_probability",
                "free_energy_kj_mol",
                "plot_x",
                "plot_y",
                "phi_center",
                "psi_center",
                "metadata",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "state_id": "cluster_0",
                "label": "cluster_0",
                "frame_count": 2,
                "reference_probability": 0.6,
                "target_probability": 0.45,
                "free_energy_kj_mol": "50.0",
                "plot_x": "-1.0",
                "plot_y": "0.5",
                "phi_center": "-60.0",
                "psi_center": "-40.0",
                "metadata": '{"reference_probability_source":"loaded"}',
            }
        )
        writer.writerow(
            {
                "state_id": "cluster_1",
                "label": "cluster_1",
                "frame_count": 1,
                "reference_probability": 0.4,
                "target_probability": 0.55,
                "free_energy_kj_mol": "80.0",
                "plot_x": "1.0",
                "plot_y": "-0.5",
                "phi_center": "50.0",
                "psi_center": "40.0",
                "metadata": '{"reference_probability_source":"loaded"}',
            }
        )

    writer = ResultsWriter(OutputConfig(output_dir=str(tmp_path / "frame_out")))
    writer.prepare_output_dir()
    writer.write_frame_results(
        [
            FrameResult(
                frame_index=0,
                stride_number=0,
                microstate_id="cluster_0",
                microstate_label="cluster_0",
                phi_degrees=-58.0,
                psi_degrees=-38.0,
                reference_energy_kj_mol=0.0,
                perturbed_energy_kj_mol=-0.5,
                delta_u_kj_mol=-0.5,
                reference_relaxed_phi_degrees=-58.0,
                reference_relaxed_psi_degrees=-38.0,
                perturbed_relaxed_phi_degrees=-58.0,
                perturbed_relaxed_psi_degrees=-38.0,
            ),
        ]
    )
    frame_csv = tmp_path / "frame_results.csv"
    frame_csv.write_text(
        Path(writer.raw_frame_results_path).read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    summary = generate_felp_plots(
        FELPPlotConfig(
            microstate_summary_csv=str(summary_csv),
            raw_frame_results_csv=str(frame_csv),
            output_dir=str(tmp_path / "felp_plots"),
            fel_max_energy_kj_mol=12.0,
            plot_energy_unit="kcal_mol",
        )
    )

    assert summary["fel_max_energy_kj_mol"] == 12.0
    assert summary["plot_energy_unit"] == "kcal_mol"
    assert (tmp_path / "felp_plots" / "delta_free_energy_projection.png").is_file()


def test_generate_felp_plots_builds_state_grid_landscapes(tmp_path: Path) -> None:
    summary_csv = tmp_path / "microstate_summary.csv"

    with open(summary_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "state_id",
                "label",
                "frame_count",
                "reference_probability",
                "target_probability",
                "free_energy_kj_mol",
                "phi_center",
                "psi_center",
                "metadata",
            ],
        )
        writer.writeheader()
        for state_id, phi, psi, pref, ptarget, free_energy in (
            ("0_0", -90.0, -90.0, 0.25, 0.10, 1.0),
            ("0_1", -90.0, 90.0, 0.25, 0.20, 0.5),
            ("1_0", 90.0, -90.0, 0.25, 0.30, 0.2),
            ("1_1", 90.0, 90.0, 0.25, 0.40, 0.0),
        ):
            writer.writerow(
                {
                    "state_id": state_id,
                    "label": state_id,
                    "frame_count": 10,
                    "reference_probability": pref,
                    "target_probability": ptarget,
                    "free_energy_kj_mol": free_energy,
                    "phi_center": phi,
                    "psi_center": psi,
                    "metadata": '{"reference_probability_source":"raw_counts"}',
                }
            )

    summary = generate_felp_plots(
        FELPPlotConfig(
            microstate_summary_csv=str(summary_csv),
            output_dir=str(tmp_path / "felp_plots"),
        )
    )

    assert summary["reference_probability_source"] == "raw_counts"
    assert (tmp_path / "felp_plots" / "reference_phi_psi_state_free_energy_landscape.png").is_file()
    assert (tmp_path / "felp_plots" / "reweighted_phi_psi_state_free_energy_landscape.png").is_file()
