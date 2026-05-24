"""Tests for JSON-driven composite Ramachandran figures."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from felp.plotting.composite import (
    CompositeColumnSpec,
    CompositeFigureConfig,
    CompositeFigureStyle,
    generate_composite_figure,
)


def _write_summary(path: Path) -> None:
    rows = [
        ("s00", -170.0, -170.0, 0.25, 0.20),
        ("s01", -170.0, -150.0, 0.25, 0.30),
        ("s10", -150.0, -170.0, 0.25, 0.15),
        ("s11", -150.0, -150.0, 0.25, 0.35),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "state_id",
                "label",
                "phi_center",
                "psi_center",
                "reference_probability",
                "target_probability",
                "free_energy_kj_mol",
                "plot_x",
                "plot_y",
            ]
        )
        for state_id, phi, psi, ref, target in rows:
            writer.writerow(
                [state_id, state_id, phi, psi, ref, target, 5.0, phi, psi]
            )


def _write_md_probability_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("plot_x", "plot_y", "probability"),
        )
        writer.writeheader()
        for phi in (-170.0, -150.0):
            for psi in (-170.0, -150.0):
                writer.writerow(
                    {
                        "plot_x": phi,
                        "plot_y": psi,
                        "probability": 0.25,
                    }
                )


def test_generate_composite_figure_writes_pdf(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    summary = tmp_path / "microstate_summary.csv"
    reference = tmp_path / "reference_summary.csv"
    _write_summary(summary)
    _write_summary(reference)
    md_csv = tmp_path / "md_phi_psi_probability.csv"
    _write_md_probability_csv(md_csv)

    output = tmp_path / "composite.pdf"
    config = CompositeFigureConfig(
        output=str(output),
        overwrite=True,
        reference={"microstate_summary": str(reference)},
        rows=(
            {
                "label": "GLY -> ALA",
                "microstate_summary": str(summary),
                "md_probability_csv": str(md_csv),
            },
        ),
        columns=(
            CompositeColumnSpec(
                type="reference_fel",
                header="Reference",
                draw_arrows_to="reweighted_fel",
            ),
            CompositeColumnSpec(type="reweighted_fel", header="FELP Reweight"),
            CompositeColumnSpec(type="md_fel", header="MD Simulation"),
            CompositeColumnSpec(type="kl_divergence", header="KL Divergence"),
        ),
        style=CompositeFigureStyle(dpi=72.0, panel_width=2.0, panel_height=2.0),
        defaults={
            "temperature_kelvin": 310.0,
            "plot_energy_unit": "kcal_mol",
            "fel_max_energy_kcal_mol": 6.0,
        },
    )
    summary_payload = generate_composite_figure(config)

    assert output.is_file()
    assert summary_payload["row_count"] == 1
    assert (tmp_path / "composite_figure_summary.json").is_file()
