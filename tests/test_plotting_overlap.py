from __future__ import annotations

import csv

import pytest

import felp.plotting.felp as felp_plotting


def test_generate_felp_plots_reports_overlap_coefficient_without_matplotlib(
    tmp_path, monkeypatch
) -> None:
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
                "free_energy_kj_mol": 0.0,
                "plot_x": "",
                "plot_y": "",
                "phi_center": "",
                "psi_center": "",
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
                "free_energy_kj_mol": 1.0,
                "plot_x": "",
                "plot_y": "",
                "phi_center": "",
                "psi_center": "",
                "metadata": '{"reference_probability_source":"loaded"}',
            }
        )

    monkeypatch.setattr(felp_plotting, "import_matplotlib", lambda: object())
    monkeypatch.setattr(felp_plotting, "apply_publication_style", lambda: None)
    monkeypatch.setattr(
        felp_plotting,
        "_plot_occupancy_shift",
        lambda **_: "occupancy_shifts.png",
    )
    monkeypatch.setattr(
        felp_plotting,
        "_plot_probability_comparison",
        lambda **_: "probability_comparison.png",
    )
    monkeypatch.setattr(
        felp_plotting,
        "_plot_kl_contributions",
        lambda **_: "kl_contributions.png",
    )
    monkeypatch.setattr(
        felp_plotting,
        "_plot_phi_psi_state_landscapes",
        lambda **_: [],
    )
    monkeypatch.setattr(
        felp_plotting,
        "_plot_discrete_microstate_ramachandran",
        lambda **_: [],
    )

    summary = felp_plotting.generate_felp_plots(
        felp_plotting.FELPPlotConfig(
            microstate_summary_csv=str(summary_csv),
            output_dir=str(tmp_path / "plots"),
        )
    )

    assert summary["overlap_coefficient"] == pytest.approx(0.85)
    assert (tmp_path / "plots" / "plot_summary.json").is_file()
