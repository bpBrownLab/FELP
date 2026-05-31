#!/usr/bin/env python3
"""Build the 3-panel T4 A99L vs wild-type psi_114 landscape figure."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

COLOR_WT = "#E8A0A0"
COLOR_WT_EDGE = "#C05050"
COLOR_MUT = "#B8A0D8"
COLOR_MUT_EDGE = "#7050A0"
KCAL_PER_KJ = 4.184
K_BOLTZMANN_KJ_PER_MOL_K = 8.314e-3


def prob_to_fe_kcal(probability: np.ndarray, temperature_kelvin: float) -> np.ndarray:
    """Convert a normalized probability profile to relative free energy (kcal/mol)."""

    kT = K_BOLTZMANN_KJ_PER_MOL_K * temperature_kelvin
    free_energy = np.full_like(probability, np.nan, dtype=float)
    occupied = probability > 0.0
    free_energy[occupied] = -kT * np.log(probability[occupied])
    free_energy[occupied] -= np.nanmin(free_energy[occupied])
    return free_energy / KCAL_PER_KJ


def load_summary(microstate_summary_csv: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = list(csv.DictReader(microstate_summary_csv.open(newline="", encoding="utf-8")))
    x_values = np.array([float(row["plot_x"]) for row in rows], dtype=float)
    reference_probability = np.array(
        [float(row["reference_probability"]) for row in rows],
        dtype=float,
    )
    target_probability = np.array(
        [float(row["target_probability"]) for row in rows],
        dtype=float,
    )
    return x_values, reference_probability, target_probability


def plot_landscape_figure(
    *,
    microstate_summary_csv: Path,
    output_png: Path,
    temperature_kelvin: float,
    fel_max_energy_kcal_mol: float,
) -> None:
    x_values, reference_probability, target_probability = load_summary(microstate_summary_csv)
    l99a_probability = reference_probability
    wild_type_probability = target_probability
    order = np.argsort(x_values)
    x_sorted = x_values[order]
    wild_type_probability = wild_type_probability[order]
    l99a_probability = l99a_probability[order]

    wild_type_fe = prob_to_fe_kcal(wild_type_probability, temperature_kelvin)
    l99a_fe = prob_to_fe_kcal(l99a_probability, temperature_kelvin)
    energy_cap_kcal = float(fel_max_energy_kcal_mol)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )

    fig, (ax_a, ax_b, ax_c) = plt.subplots(3, 1, figsize=(3.5, 5.5), sharex=True)

    probability_ymax = max(float(l99a_probability.max()), float(wild_type_probability.max()))
    probability_ylim = (0.0, probability_ymax * 1.08)

    ax_a.fill_between(
        x_sorted,
        0.0,
        l99a_probability,
        step="mid",
        alpha=0.5,
        color=COLOR_MUT,
        linewidth=0.0,
    )
    ax_a.step(
        x_sorted,
        l99a_probability,
        where="mid",
        color=COLOR_MUT_EDGE,
        linewidth=1.2,
    )
    ax_a.set_ylabel("Probability")
    ax_a.set_ylim(probability_ylim)
    ax_a.set_title("L99A", fontsize=9, fontweight="bold", pad=4)
    ax_a.text(-0.15, 1.05, "A", transform=ax_a.transAxes, fontsize=11, fontweight="bold", va="top")

    ax_b.fill_between(
        x_sorted,
        0.0,
        wild_type_probability,
        step="mid",
        alpha=0.5,
        color=COLOR_WT,
        linewidth=0.0,
    )
    ax_b.step(
        x_sorted,
        wild_type_probability,
        where="mid",
        color=COLOR_WT_EDGE,
        linewidth=1.2,
    )
    ax_b.set_ylabel("Probability")
    ax_b.set_ylim(probability_ylim)
    ax_b.set_title("Wild-Type (L99A --> WT)", fontsize=9, fontweight="bold", pad=4)
    ax_b.text(-0.15, 1.05, "B", transform=ax_b.transAxes, fontsize=11, fontweight="bold", va="top")

    wild_type_fe_plot = np.where(wild_type_fe <= energy_cap_kcal, wild_type_fe, np.nan)
    l99a_fe_plot = np.where(l99a_fe <= energy_cap_kcal, l99a_fe, np.nan)
    ax_c.step(
        x_sorted,
        l99a_fe_plot,
        where="mid",
        color=COLOR_MUT_EDGE,
        linewidth=1.4,
        label="L99A",
    )
    ax_c.step(
        x_sorted,
        wild_type_fe_plot,
        where="mid",
        color=COLOR_WT_EDGE,
        linewidth=1.4,
        label="Wild-Type",
    )
    ax_c.axhline(energy_cap_kcal, color="0.75", linewidth=0.5, linestyle="--")
    ax_c.set_ylabel(r"$\Delta G$ (kcal mol$^{-1}$)")
    ax_c.set_xlabel(r"$\psi_{114}$ (deg)")
    ax_c.set_ylim(-0.3, energy_cap_kcal * 1.06)
    ax_c.set_title("Free energy landscape", fontsize=9, fontweight="bold", pad=4)
    ax_c.text(-0.15, 1.05, "C", transform=ax_c.transAxes, fontsize=11, fontweight="bold", va="top")
    ax_c.legend(
        fontsize=7,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=2,
        borderaxespad=0.0,
    )

    for axis in (ax_a, ax_b, ax_c):
        axis.set_xlim(-180.0, 180.0)
        axis.xaxis.set_minor_locator(AutoMinorLocator())
        axis.yaxis.set_minor_locator(AutoMinorLocator())

    fig.tight_layout(h_pad=1.2)
    fig.subplots_adjust(bottom=0.12)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output_png.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--microstate-summary",
        required=True,
        type=Path,
        help="Path to microstate_summary.csv from felp summarize.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output PNG path (PDF written alongside).",
    )
    parser.add_argument("--temperature-kelvin", type=float, default=310.0)
    parser.add_argument(
        "--fel-max-energy-kcal-mol",
        type=float,
        default=8.0,
        help="Free-energy display ceiling in kcal/mol (default: 8.0).",
    )
    args = parser.parse_args()

    plot_landscape_figure(
        microstate_summary_csv=args.microstate_summary,
        output_png=args.output,
        temperature_kelvin=args.temperature_kelvin,
        fel_max_energy_kcal_mol=args.fel_max_energy_kcal_mol,
    )
    print(args.output)
    print(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
