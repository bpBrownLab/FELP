"""Tests for neighbor-pool-aware frame projection in md-kl-plot."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from felp.core import FrameResult
from felp.plotting.md_kl import (
    MDKLPlotConfig,
    _load_frame_projected_reweighted_surface,
    generate_md_kl_plot,
)


def _frame(frame_index: int, state_id: str, *, phi: float, psi: float, delta_u: float) -> FrameResult:
    return FrameResult(
        frame_index=frame_index,
        stride_number=frame_index,
        microstate_id=state_id,
        microstate_label=state_id,
        phi_degrees=phi,
        psi_degrees=psi,
        reference_energy_kj_mol=0.0,
        perturbed_energy_kj_mol=delta_u,
        delta_u_kj_mol=delta_u,
        reference_relaxed_phi_degrees=phi,
        reference_relaxed_psi_degrees=psi,
        perturbed_relaxed_phi_degrees=phi,
        perturbed_relaxed_psi_degrees=psi,
    )


def test_md_kl_frame_projection_passes_neighbor_pool_factors(tmp_path: Path) -> None:
    summary_csv = tmp_path / "microstate_summary.csv"
    summary_csv.write_text(
        "state_id,label,reference_probability,target_probability,reweight_factor,metadata\n"
        's0,s0,0.6,0.55,1.2,"{""reweighting_method"":""neighbor_pool""}"\n'
        's1,s1,0.4,0.45,0.9,"{""reweighting_method"":""neighbor_pool""}"\n',
        encoding="utf-8",
    )
    frame_csv = tmp_path / "frame_results.csv"
    frame_csv.write_text(
        "frame_index,stride_number,microstate_id,microstate_label,phi_degrees,psi_degrees,"
        "reference_energy_kj_mol,perturbed_energy_kj_mol,delta_u_kj_mol,"
        "reference_relaxed_phi_degrees,reference_relaxed_psi_degrees,"
        "perturbed_relaxed_phi_degrees,perturbed_relaxed_psi_degrees\n"
        "0,0,s0,s0,-60,120,0,0,0,-60,120,-60,120\n"
        "1,1,s1,s1,60,-30,0,0,1,60,-30,60,-30\n",
        encoding="utf-8",
    )
    config = MDKLPlotConfig(
        output_dir=str(tmp_path / "plots"),
        topology_file=str(tmp_path / "top.pdb"),
        trajectory_files=(("traj.nc",),),
        reweighted_surface_csv=str(summary_csv),
        raw_frame_results_csv=str(frame_csv),
        phi_psi_residue_number=1,
        fel_bins=36,
        fel_max_points=10_000,
    )

    captured: dict[str, object] = {}

    def _capture_collect(**kwargs: object) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        captured.update(kwargs)
        x = np.asarray([-60.0, 60.0])
        y = np.asarray([120.0, -30.0])
        ref = np.asarray([0.6, 0.4])
        target = np.asarray([0.33, 0.27])
        return x, y, ref, target

    with patch(
        "felp.plotting.md_kl._collect_cv_projection_samples",
        side_effect=_capture_collect,
    ):
        surface, x_edges, y_edges, metadata = _load_frame_projected_reweighted_surface(config)

    assert captured["reweighting_method"] == "neighbor_pool"
    assert captured["state_reweight_factors"] == {"s0": 1.2, "s1": 0.9}
    assert captured["frame_weights_by_index"] is None
    assert metadata["reweighting_method"] == "neighbor_pool"
    assert metadata["reweighted_frame_projection_state_mass_preserved"] is True
    assert "neighbor_pool" in str(metadata["reweighted_frame_projection_estimator_detail"])
    assert surface.sum() == pytest.approx(1.0)
    assert x_edges.size > 1 and y_edges.size > 1


def _write_complete_2x2_summary(path: Path) -> None:
    rows = [
        ("s00", -170.0, -170.0, 0.25),
        ("s01", -170.0, -150.0, 0.25),
        ("s10", -150.0, -170.0, 0.25),
        ("s11", -150.0, -150.0, 0.25),
    ]
    lines = [
        "state_id,label,phi_center,psi_center,target_probability,reference_probability"
    ]
    for state_id, phi, psi, probability in rows:
        lines.append(
            f"{state_id},{state_id},{phi},{psi},{probability},{probability}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_md_kl_prefers_state_summary_probabilities_for_kl(tmp_path: Path) -> None:
    summary_csv = tmp_path / "microstate_summary.csv"
    _write_complete_2x2_summary(summary_csv)
    frame_csv = tmp_path / "frame_results.csv"
    frame_csv.write_text("frame_index\n0\n", encoding="utf-8")
    topology = tmp_path / "top.pdb"
    topology.write_text("END\n", encoding="utf-8")
    trajectory = tmp_path / "traj.nc"
    trajectory.write_text("", encoding="utf-8")
    config = MDKLPlotConfig(
        output_dir=str(tmp_path / "plots"),
        topology_file=str(topology),
        trajectory_files=((str(trajectory),),),
        reweighted_surface_csv=str(summary_csv),
        raw_frame_results_csv=str(frame_csv),
        phi_psi_residue_number=1,
        fel_bins=10,
        fel_max_points=10_000,
        overwrite=True,
    )

    def _reject_frame_projection(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("frame projection must not be used when the state grid is complete")

    md_phi = np.full(16, -160.0)
    md_psi = np.full(16, -160.0)
    with patch(
        "felp.plotting.md_kl._load_frame_projected_reweighted_surface",
        side_effect=_reject_frame_projection,
    ), patch(
        "felp.plotting.md_kl._load_md_phi_psi_degrees",
        return_value=(md_phi, md_psi),
    ):
        result = generate_md_kl_plot(config)

    assert result["raw_frame_results_used_for_kl"] is False
    assert result["reweighted_kl_probability_surface_source"] == "regular_surface_csv"
    assert result["surface_shape"] == [2, 2]
