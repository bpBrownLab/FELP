from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from felp.core import ConfigurationError, FrameResult, MicrostateDefinition, OutputConfig
from felp.io import ResultsWriter
from felp.msm import MSMBundleManifest, save_bundle
from felp.plotting import FELPPlotConfig, MSMPlotConfig, generate_felp_plots, generate_msm_plots
from felp.plotting.felp import (
    _collect_cv_projection_samples,
    _excluded_frame_indices_by_state,
    _filter_qc_excluded_frame_results,
    _gaussian_kernel_projected_surfaces,
    _normalize_cv_name,
    _resolve_cv_axes,
    _state_kernel_probability_grid,
)


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
    assert summary["overlap_coefficient"] == pytest.approx(0.85)
    assert summary["fel_colormap"] == "nipy_spectral_r"
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


def test_generate_felp_plots_supports_custom_cv_axes(tmp_path: Path) -> None:
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
                "metadata",
            ],
        )
        writer.writeheader()
        for state_id, ends, turn, pref, ptarget, free_energy in (
            ("short_tight", 10.0, 4.0, 0.25, 0.10, 1.0),
            ("short_open", 10.0, 8.0, 0.25, 0.20, 0.5),
            ("long_tight", 20.0, 4.0, 0.25, 0.30, 0.2),
            ("long_open", 20.0, 8.0, 0.25, 0.40, 0.0),
        ):
            writer.writerow(
                {
                    "state_id": state_id,
                    "label": state_id,
                    "frame_count": 1,
                    "reference_probability": pref,
                    "target_probability": ptarget,
                    "free_energy_kj_mol": free_energy,
                    "metadata": (
                        '{"reference_probability_source":"raw_counts",'
                        f'"cv_centers":{{"ends":{ends},"turn":{turn}}},'
                        f'"cv_ranges":{{"ends":[{ends - 1.0},{ends + 1.0}],'
                        f'"turn":[{turn - 0.5},{turn + 0.5}]}}'
                        "}"
                    ),
                }
            )

    writer = ResultsWriter(OutputConfig(output_dir=str(tmp_path / "frame_out")))
    writer.prepare_output_dir()
    writer.write_frame_results(
        [
            FrameResult(
                frame_index=index,
                stride_number=index,
                microstate_id=state_id,
                microstate_label=state_id,
                phi_degrees=None,
                psi_degrees=None,
                reference_energy_kj_mol=0.0,
                perturbed_energy_kj_mol=delta_u,
                delta_u_kj_mol=delta_u,
                reference_relaxed_phi_degrees=None,
                reference_relaxed_psi_degrees=None,
                perturbed_relaxed_phi_degrees=None,
                perturbed_relaxed_psi_degrees=None,
                metadata={
                    "assignment_measurements": {
                        "ends": ends,
                        "turn": turn,
                    }
                },
            )
            for index, (state_id, ends, turn, delta_u) in enumerate(
                (
                    ("short_tight", 10.1, 4.1, -0.2),
                    ("short_open", 9.9, 7.9, 0.1),
                    ("long_tight", 20.2, 4.2, 0.3),
                    ("long_open", 19.8, 7.8, -0.4),
                )
            )
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
            cv_x="ends",
            cv_y="turn",
            fel_bins=20,
            weight_treatment="psis",
        )
    )

    assert summary["cv_axes"] == ["ends", "turn"]
    assert summary["cv_frame_projection_available"] is True
    assert summary["cv_frame_projection_estimator"] == "histogram"
    assert summary["cv_frame_projection_weight_treatment"] == "psis-smoothed"
    assert summary["cv_frame_projection_weight_treatment_source"] == "cli_override"
    assert summary["cv_frame_projection_state_mass_preserved"] is False
    assert summary["state_reweighting_weight_treatment"] == "raw"
    assert "phi_psi_frame_projection_available" not in summary
    assert (tmp_path / "felp_plots" / "reference_ends_turn_state_free_energy_landscape.png").is_file()
    assert (tmp_path / "felp_plots" / "reweighted_ends_turn_state_free_energy_landscape.png").is_file()
    assert (tmp_path / "felp_plots" / "reference_ends_turn_free_energy_landscape.png").is_file()
    assert (tmp_path / "felp_plots" / "reweighted_ends_turn_free_energy_landscape.png").is_file()
    assert (tmp_path / "felp_plots" / "discrete_microstate_free_energy_ends_turn.png").is_file()
    assert not (tmp_path / "felp_plots" / "reference_probability_projection.png").is_file()


def test_frame_projection_filters_qc_excluded_frames() -> None:
    rows = [
        {
            "state_id": "state_a",
            "label": "state_a",
            "metadata": '{"excluded_frame_indices":[0]}',
        }
    ]
    frame_results = [
        FrameResult(
            frame_index=0,
            stride_number=0,
            microstate_id="state_a",
            microstate_label="state_a",
            phi_degrees=None,
            psi_degrees=None,
            reference_energy_kj_mol=0.0,
            perturbed_energy_kj_mol=-100.0,
            delta_u_kj_mol=-100.0,
            reference_relaxed_phi_degrees=None,
            reference_relaxed_psi_degrees=None,
            perturbed_relaxed_phi_degrees=None,
            perturbed_relaxed_psi_degrees=None,
            metadata={"assignment_measurements": {"ends": 10.0, "turn": 4.0}},
        ),
        FrameResult(
            frame_index=1,
            stride_number=1,
            microstate_id="state_a",
            microstate_label="state_a",
            phi_degrees=None,
            psi_degrees=None,
            reference_energy_kj_mol=0.0,
            perturbed_energy_kj_mol=0.0,
            delta_u_kj_mol=0.0,
            reference_relaxed_phi_degrees=None,
            reference_relaxed_psi_degrees=None,
            perturbed_relaxed_phi_degrees=None,
            perturbed_relaxed_psi_degrees=None,
            metadata={"assignment_measurements": {"ends": 20.0, "turn": 8.0}},
        ),
    ]

    excluded = _excluded_frame_indices_by_state(rows)
    filtered = _filter_qc_excluded_frame_results(frame_results, excluded)
    sampled = _collect_cv_projection_samples(
        frame_results=filtered,
        reference_probability_by_state={"state_a": 1.0},
        temperature_kelvin=300.0,
        weight_treatment="raw",
        max_points=10,
        x_name="ends",
        y_name="turn",
    )

    assert [result.frame_index for result in filtered] == [1]
    assert sampled is not None
    x_values, y_values, _, _ = sampled
    assert x_values.tolist() == [20.0]
    assert y_values.tolist() == [8.0]


def test_frame_projection_uses_psis_smoothed_weights() -> None:
    frame_results = [
        FrameResult(
            frame_index=index,
            stride_number=index,
            microstate_id="state_a",
            microstate_label="state_a",
            phi_degrees=None,
            psi_degrees=None,
            reference_energy_kj_mol=0.0,
            perturbed_energy_kj_mol=delta_u,
            delta_u_kj_mol=delta_u,
            reference_relaxed_phi_degrees=None,
            reference_relaxed_psi_degrees=None,
            perturbed_relaxed_phi_degrees=None,
            perturbed_relaxed_psi_degrees=None,
            metadata={
                "assignment_measurements": {
                    "ends": float(index),
                    "turn": float(index % 3),
                }
            },
        )
        for index, delta_u in enumerate([0.0] * 29 + [-10.0])
    ]

    raw = _collect_cv_projection_samples(
        frame_results=frame_results,
        reference_probability_by_state={"state_a": 1.0},
        temperature_kelvin=300.0,
        weight_treatment="raw",
        max_points=100,
        x_name="ends",
        y_name="turn",
    )
    smoothed = _collect_cv_projection_samples(
        frame_results=frame_results,
        reference_probability_by_state={"state_a": 1.0},
        temperature_kelvin=300.0,
        weight_treatment="psis-smoothed",
        max_points=100,
        x_name="ends",
        y_name="turn",
    )

    assert raw is not None
    assert smoothed is not None
    assert smoothed[3].max() < raw[3].max()


def test_state_kernel_probability_grid_preserves_finite_grid_mass() -> None:
    grid = _state_kernel_probability_grid(
        x_values=np.asarray([0.0]),
        y_values=np.asarray([0.0]),
        weights=np.asarray([1.0]),
        x_centers=np.linspace(0.0, 4.0, 5),
        y_centers=np.linspace(0.0, 4.0, 5),
        bandwidth=(1.0, 1.0),
    )

    assert grid.sum() == pytest.approx(1.0)


def test_gaussian_kernel_projection_preserves_probability_mass() -> None:
    frame_results = [
        FrameResult(
            frame_index=index,
            stride_number=index,
            microstate_id=state_id,
            microstate_label=state_id,
            phi_degrees=None,
            psi_degrees=None,
            reference_energy_kj_mol=0.0,
            perturbed_energy_kj_mol=delta_u,
            delta_u_kj_mol=delta_u,
            reference_relaxed_phi_degrees=None,
            reference_relaxed_psi_degrees=None,
            perturbed_relaxed_phi_degrees=None,
            perturbed_relaxed_psi_degrees=None,
            metadata={"assignment_measurements": {"ends": ends, "turn": turn}},
        )
        for index, (state_id, ends, turn, delta_u) in enumerate(
            (
                ("a", 0.0, 0.0, 0.0),
                ("a", 0.3, 0.2, 1.0),
                ("b", 4.0, 4.0, 0.0),
                ("b", 4.2, 4.1, -1.0),
            )
        )
    ]

    result = _gaussian_kernel_projected_surfaces(
        frame_results=frame_results,
        reference_probability_by_state={"a": 0.25, "b": 0.75},
        target_probability_by_state={"a": 0.10, "b": 0.90},
        temperature_kelvin=300.0,
        weight_treatment="raw",
        bins=20,
        max_points=100,
        x_name="ends",
        y_name="turn",
        bandwidth_spec="0.5,0.5",
        bandwidth_rule="scott",
    )

    assert result is not None
    reference_grid, target_grid, _, _, metadata = result
    assert reference_grid.sum() == pytest.approx(1.0)
    assert target_grid.sum() == pytest.approx(1.0)
    assert metadata["cv_frame_projection_state_mass_preserved"] is True
    assert metadata["cv_frame_projection_bandwidth"] == [0.5, 0.5]


def test_gaussian_kernel_projection_spreads_sparse_weighted_surface() -> None:
    frame_results = [
        FrameResult(
            frame_index=index,
            stride_number=index,
            microstate_id="a",
            microstate_label="a",
            phi_degrees=None,
            psi_degrees=None,
            reference_energy_kj_mol=0.0,
            perturbed_energy_kj_mol=0.0,
            delta_u_kj_mol=0.0,
            reference_relaxed_phi_degrees=None,
            reference_relaxed_psi_degrees=None,
            perturbed_relaxed_phi_degrees=None,
            perturbed_relaxed_psi_degrees=None,
            metadata={"assignment_measurements": {"ends": ends, "turn": turn}},
        )
        for index, (ends, turn) in enumerate(((0.0, 0.0), (5.0, 5.0)))
    ]

    result = _gaussian_kernel_projected_surfaces(
        frame_results=frame_results,
        reference_probability_by_state={"a": 1.0},
        target_probability_by_state={"a": 1.0},
        temperature_kelvin=300.0,
        weight_treatment="raw",
        bins=30,
        max_points=100,
        x_name="ends",
        y_name="turn",
        bandwidth_spec="1.0",
        bandwidth_rule="scott",
    )

    assert result is not None
    _, target_grid, _, _, _ = result
    assert int(np.count_nonzero(target_grid > 1.0e-6)) > 2


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("ends", "ends"),
        ("turn", "turn"),
        ("ends:1:CA:10:CA", "ends"),
        ("turn:3:CA:6:CA", "turn"),
        ("phi", "phi"),
        ("a:b", "a:b"),
    ],
)
def test_normalize_cv_name(raw: str, expected: str) -> None:
    assert _normalize_cv_name(raw) == expected


def test_felp_plot_config_normalizes_spec_strings(tmp_path: Path) -> None:
    csv_file = tmp_path / "summary.csv"
    csv_file.write_text("label,reference_probability,target_probability\n", encoding="utf-8")
    config = FELPPlotConfig(
        microstate_summary_csv=str(csv_file),
        output_dir=str(tmp_path / "plots"),
        cv_x="ends:1:CA:10:CA",
        cv_y="turn:3:CA:6:CA",
    )
    config.validate()
    assert config.cv_x == "ends"
    assert config.cv_y == "turn"


def test_felp_plot_config_rejects_unknown_colormap(tmp_path: Path) -> None:
    csv_file = tmp_path / "summary.csv"
    csv_file.write_text("label,reference_probability,target_probability\n", encoding="utf-8")
    config = FELPPlotConfig(
        microstate_summary_csv=str(csv_file),
        output_dir=str(tmp_path / "plots"),
        fel_colormap="not_a_valid_matplotlib_cmap_name_xyz",
    )
    with pytest.raises(ConfigurationError):
        config.validate()


def test_resolve_cv_axes_prefers_custom_cvs_over_phi_psi() -> None:
    rows = [
        {
            "metadata": (
                '{"cv_centers":{"phi":-60,"psi":150,"ends":15.0,"turn":6.0},'
                '"cv_ranges":{"phi":[-70,-50],"psi":[140,160],'
                '"ends":[14,16],"turn":[5,7]}}'
            ),
        }
    ]
    axes = _resolve_cv_axes(
        rows=rows, frame_results=None, requested_x=None, requested_y=None,
    )
    assert axes is not None
    assert axes.x_name == "ends"
    assert axes.y_name == "turn"
    assert axes.source == "inferred_custom_cv"


def test_resolve_cv_axes_falls_back_to_phi_psi() -> None:
    rows = [
        {
            "metadata": (
                '{"cv_centers":{"phi":-60,"psi":150},'
                '"cv_ranges":{"phi":[-70,-50],"psi":[140,160]}}'
            ),
        }
    ]
    axes = _resolve_cv_axes(
        rows=rows, frame_results=None, requested_x=None, requested_y=None,
    )
    assert axes is not None
    assert axes.x_name == "phi"
    assert axes.y_name == "psi"
    assert axes.source == "inferred_phi_psi"
