#!/usr/bin/env python3
"""Six-panel chignolin figure: MD/FELP landscapes + folding free energies."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from felp.plotting.common import free_energy_from_probabilities, scatter_state_projection
from scripts.chignolin_folding_free_energy_compare import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    BasinSpec,
    BootstrapEstimate,
    bootstrap_summary,
    delta_g_fold_rest,
    experiment_consensus_point,
    folded_mask,
    load_felp_frame_arrays,
    point_estimate_bootstrap,
    weighted_folded_fraction,
)

RECOMMENDED_FOLDED = BasinSpec(3.0, 7.0, 5.0, 6.5)
DEFAULT_FEL_MAX_KCAL_MOL = 6.0
FEL_COLORMAP = "gist_earth"
FEL_XLABEL = r"C$\alpha_1$–C$\alpha_{10}$ distance (Å)"
FEL_YLABEL = r"C$\alpha_3$–C$\alpha_6$ distance (Å)"
PANEL_E_VARIANT_COLORS = {
    "WT": (0.93, 0.72, 0.74, 0.75),
    "CLN025": (0.80, 0.66, 0.82, 0.75),
}
PANEL_F_BAR_COLOR = (0.84, 0.94, 0.84, 0.75)
BAR_EDGE_COLOR = "black"
BAR_EDGE_WIDTH = 0.7
FOLDED_BOX_COLOR = "#111111"
BOOTSTRAP_CACHE_VERSION = 1

FONT_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 18,
    "axes.labelsize": 18,
    "axes.titlesize": 19,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def load_microstate_arrays(
    summary_csv: str,
    *,
    probability_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_values: list[float] = []
    y_values: list[float] = []
    probabilities: list[float] = []
    with open(summary_csv, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            x_values.append(float(row["plot_x"]))
            y_values.append(float(row["plot_y"]))
            probabilities.append(float(row[probability_column]))
    return (
        np.asarray(x_values, dtype=float),
        np.asarray(y_values, dtype=float),
        np.asarray(probabilities, dtype=float),
    )


def draw_fel_panel(
    ax,
    summary_csv: str,
    *,
    probability_column: str,
    title: str,
    temperature_kelvin: float,
    folded: BasinSpec,
    fel_max_energy_kcal_mol: float,
    panel_label: str,
) -> None:
    x, y, probabilities = load_microstate_arrays(
        summary_csv,
        probability_column=probability_column,
    )
    free_energy_kj_mol = free_energy_from_probabilities(
        probabilities,
        temperature_kelvin=temperature_kelvin,
    )
    free_energy_kcal_mol = free_energy_kj_mol / 4.184

    scatter_state_projection(
        ax,
        x,
        y,
        free_energy_kcal_mol,
        colorbar_label=r"Relative free energy (kcal mol$^{-1}$)",
        title=title,
        xlabel=FEL_XLABEL,
        ylabel=FEL_YLABEL,
        fel_max_energy_kj_mol=fel_max_energy_kcal_mol,
        colormap=FEL_COLORMAP,
        show_colorbar=False,
        marker_size=48.0,
        ramachandran_equal_aspect=False,
    )
    from matplotlib.patches import Rectangle

    rect = Rectangle(
        (folded.ends_min, folded.turn_min),
        folded.ends_max - folded.ends_min,
        folded.turn_max - folded.turn_min,
        linewidth=2.4,
        edgecolor=FOLDED_BOX_COLOR,
        facecolor="none",
        linestyle="--",
        zorder=5,
    )
    ax.add_patch(rect)
    ax.text(
        -0.10,
        1.08,
        panel_label,
        transform=ax.transAxes,
        fontsize=24,
        fontweight="bold",
        va="bottom",
        ha="right",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _input_metadata(*paths: str) -> dict[str, dict[str, float | int]]:
    metadata: dict[str, dict[str, float | int]] = {}
    for path in paths:
        stat = os.stat(path)
        metadata[path] = {
            "mtime_ns": int(stat.st_mtime_ns),
            "size": int(stat.st_size),
        }
    return metadata


def _bootstrap_cache_path(output_path: str, cache_path: str | None) -> str:
    if cache_path:
        return cache_path
    return os.path.splitext(output_path)[0] + "_bootstrap_cache.json"


def _folded_basin_dict(folded: BasinSpec) -> dict[str, float]:
    return {
        "ends_min": folded.ends_min,
        "ends_max": folded.ends_max,
        "turn_min": folded.turn_min,
        "turn_max": folded.turn_max,
    }


def _cache_is_valid(
    cached: dict[str, object],
    *,
    folded: BasinSpec,
    temperature_kelvin: float,
    n_bootstrap: int,
    bootstrap_seed: int,
    input_metadata: dict[str, dict[str, float | int]],
) -> bool:
    if cached.get("cache_version") != BOOTSTRAP_CACHE_VERSION:
        return False
    if cached.get("folded_basin") != _folded_basin_dict(folded):
        return False
    if cached.get("temperature_kelvin") != temperature_kelvin:
        return False
    if cached.get("bootstrap_samples") != n_bootstrap:
        return False
    if cached.get("bootstrap_seed") != bootstrap_seed:
        return False
    return cached.get("input_metadata") == input_metadata


def _md_bootstrap_dg_samples(
    distances_csv: str,
    folded: BasinSpec,
    *,
    temperature_kelvin: float,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, BootstrapEstimate, np.ndarray]:
    data = np.loadtxt(distances_csv, skiprows=1)
    is_folded = folded_mask(data[:, 1], data[:, 2], folded)
    p_point = float(np.mean(is_folded))
    rng = np.random.default_rng(seed)
    n_frames = len(is_folded)
    dg_samples = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sample = is_folded[rng.integers(0, n_frames, size=n_frames)]
        dg_samples[index] = delta_g_fold_rest(float(np.mean(sample)), temperature_kelvin)
    return p_point, bootstrap_summary(dg_samples), dg_samples


def _felp_bootstrap_dg_samples(
    frame_results_csv: str,
    summary_csv: str,
    folded: BasinSpec,
    *,
    temperature_kelvin: float,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, BootstrapEstimate, np.ndarray]:
    is_folded, weights = load_felp_frame_arrays(frame_results_csv, summary_csv, folded)
    p_point = weighted_folded_fraction(is_folded, weights)
    rng = np.random.default_rng(seed)
    n_frames = len(is_folded)
    dg_samples = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sample_idx = rng.integers(0, n_frames, size=n_frames)
        p_boot = weighted_folded_fraction(is_folded[sample_idx], weights[sample_idx])
        dg_samples[index] = delta_g_fold_rest(p_boot, temperature_kelvin)
    return p_point, bootstrap_summary(dg_samples), dg_samples


def _experiment_dg_point(
    *,
    variant: str,
    target_temperature_kelvin: float,
) -> BootstrapEstimate:
    _, _, dg_boot = experiment_consensus_point(
        variant=variant,
        target_temperature_kelvin=target_temperature_kelvin,
    )
    return dg_boot


def _variant_entry(dg_boot: BootstrapEstimate) -> dict[str, float]:
    return {
        "delta_g": dg_boot.mean,
        "delta_g_ci_low": dg_boot.ci_low,
        "delta_g_ci_high": dg_boot.ci_high,
        "delta_g_std": dg_boot.std,
    }


def _dd_entry(boot: BootstrapEstimate) -> dict[str, float]:
    return {
        "mean": boot.mean,
        "ci_low": boot.ci_low,
        "ci_high": boot.ci_high,
        "std": boot.std,
    }


def build_thermodynamics(
    *,
    folded: BasinSpec,
    temperature_kelvin: float,
    wt_md_distances: str,
    cln_md_distances: str,
    wt_to_cln_summary: str,
    cln_to_wt_summary: str,
    wt_to_cln_frame_results: str,
    cln_to_wt_frame_results: str,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    seed = bootstrap_seed
    sample_arrays: dict[str, np.ndarray] = {}

    wt_exp_dg = _experiment_dg_point(
        variant="wild-type", target_temperature_kelvin=temperature_kelvin,
    )
    cln_exp_dg = _experiment_dg_point(
        variant="CLN025", target_temperature_kelvin=temperature_kelvin,
    )
    dd_boot: dict[str, BootstrapEstimate] = {
        "experiment": point_estimate_bootstrap(cln_exp_dg.mean - wt_exp_dg.mean),
    }

    if n_bootstrap <= 0:
        from scripts.chignolin_folding_free_energy_compare import (
            felp_fold_rest_from_summary,
            md_fold_rest_point,
        )

        wt_md_p, _, wt_md_dg, _ = md_fold_rest_point(
            wt_md_distances, folded, temperature_kelvin=temperature_kelvin,
        )
        cln_md_p, _, cln_md_dg, _ = md_fold_rest_point(
            cln_md_distances, folded, temperature_kelvin=temperature_kelvin,
        )
        _, _, felp_cln_dg, _ = felp_fold_rest_from_summary(
            wt_to_cln_summary, folded,
            probability_column="target_probability",
            temperature_kelvin=temperature_kelvin,
        )
        _, _, felp_wt_dg, _ = felp_fold_rest_from_summary(
            cln_to_wt_summary, folded,
            probability_column="target_probability",
            temperature_kelvin=temperature_kelvin,
        )
        dd_boot["md"] = point_estimate_bootstrap(cln_md_dg.mean - wt_md_dg.mean)
        dd_boot["felp"] = point_estimate_bootstrap(felp_cln_dg.mean - felp_wt_dg.mean)
    else:
        wt_md_p, wt_md_dg, wt_md_samples = _md_bootstrap_dg_samples(
            wt_md_distances, folded, temperature_kelvin=temperature_kelvin,
            n_bootstrap=n_bootstrap, seed=seed + 1,
        )
        cln_md_p, cln_md_dg, cln_md_samples = _md_bootstrap_dg_samples(
            cln_md_distances, folded, temperature_kelvin=temperature_kelvin,
            n_bootstrap=n_bootstrap, seed=seed + 3,
        )
        _, felp_cln_dg, felp_cln_samples = _felp_bootstrap_dg_samples(
            wt_to_cln_frame_results, wt_to_cln_summary, folded,
            temperature_kelvin=temperature_kelvin, n_bootstrap=n_bootstrap, seed=seed + 11,
        )
        _, felp_wt_dg, felp_wt_samples = _felp_bootstrap_dg_samples(
            cln_to_wt_frame_results, cln_to_wt_summary, folded,
            temperature_kelvin=temperature_kelvin, n_bootstrap=n_bootstrap, seed=seed + 12,
        )
        sample_arrays = {
            "md_wt_delta_g_kcal_mol": wt_md_samples,
            "md_cln025_delta_g_kcal_mol": cln_md_samples,
            "felp_wt_delta_g_kcal_mol": felp_wt_samples,
            "felp_cln025_delta_g_kcal_mol": felp_cln_samples,
        }
        dd_boot["md"] = bootstrap_summary(cln_md_samples - wt_md_samples)
        dd_boot["felp"] = bootstrap_summary(felp_cln_samples - felp_wt_samples)

    thermo = {
        "experiment": {
            "WT": _variant_entry(wt_exp_dg),
            "CLN025": _variant_entry(cln_exp_dg),
        },
        "md": {
            "WT": {**_variant_entry(wt_md_dg), "p_fold": wt_md_p},
            "CLN025": {**_variant_entry(cln_md_dg), "p_fold": cln_md_p},
        },
        "felp": {
            "WT": _variant_entry(felp_wt_dg),
            "CLN025": _variant_entry(felp_cln_dg),
        },
        "delta_delta_g": {
            method: _dd_entry(boot)
            for method, boot in dd_boot.items()
        },
    }
    return thermo, sample_arrays


def load_or_build_thermodynamics(
    *,
    folded: BasinSpec,
    temperature_kelvin: float,
    wt_md_distances: str,
    cln_md_distances: str,
    wt_to_cln_summary: str,
    cln_to_wt_summary: str,
    wt_to_cln_frame_results: str,
    cln_to_wt_frame_results: str,
    n_bootstrap: int,
    bootstrap_seed: int,
    cache_path: str,
    refresh_bootstrap: bool,
) -> dict[str, object]:
    input_paths = (
        wt_md_distances,
        cln_md_distances,
        wt_to_cln_summary,
        cln_to_wt_summary,
        wt_to_cln_frame_results,
        cln_to_wt_frame_results,
    )
    input_metadata = _input_metadata(*input_paths)

    if not refresh_bootstrap and os.path.isfile(cache_path):
        with open(cache_path, encoding="utf-8") as handle:
            cached = json.load(handle)
        if _cache_is_valid(
            cached,
            folded=folded,
            temperature_kelvin=temperature_kelvin,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
            input_metadata=input_metadata,
        ):
            return cached["thermodynamics"]

    thermo, sample_arrays = build_thermodynamics(
        folded=folded,
        temperature_kelvin=temperature_kelvin,
        wt_md_distances=wt_md_distances,
        cln_md_distances=cln_md_distances,
        wt_to_cln_summary=wt_to_cln_summary,
        cln_to_wt_summary=cln_to_wt_summary,
        wt_to_cln_frame_results=wt_to_cln_frame_results,
        cln_to_wt_frame_results=cln_to_wt_frame_results,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
    )

    cache_payload = {
        "cache_version": BOOTSTRAP_CACHE_VERSION,
        "folded_basin": _folded_basin_dict(folded),
        "temperature_kelvin": temperature_kelvin,
        "bootstrap_samples": n_bootstrap,
        "bootstrap_seed": bootstrap_seed,
        "input_metadata": input_metadata,
        "thermodynamics": thermo,
        "bootstrap_sample_arrays": {
            key: values.tolist() for key, values in sample_arrays.items()
        },
    }
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as handle:
        json.dump(cache_payload, handle, indent=2)
    return thermo


def plot_figure(
    output_path: str,
    *,
    folded: BasinSpec,
    temperature_kelvin: float,
    wt_to_cln_summary: str,
    cln_to_wt_summary: str,
    wt_md_distances: str,
    cln_md_distances: str,
    wt_to_cln_frame_results: str,
    cln_to_wt_frame_results: str,
    fel_max_energy_kcal_mol: float,
    n_bootstrap: int,
    bootstrap_seed: int,
    bootstrap_cache_path: str | None,
    refresh_bootstrap: bool,
) -> dict[str, object]:
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams.update(FONT_RC)

    cache_path = _bootstrap_cache_path(output_path, bootstrap_cache_path)
    thermo = load_or_build_thermodynamics(
        folded=folded,
        temperature_kelvin=temperature_kelvin,
        wt_md_distances=wt_md_distances,
        cln_md_distances=cln_md_distances,
        wt_to_cln_summary=wt_to_cln_summary,
        cln_to_wt_summary=cln_to_wt_summary,
        wt_to_cln_frame_results=wt_to_cln_frame_results,
        cln_to_wt_frame_results=cln_to_wt_frame_results,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
        cache_path=cache_path,
        refresh_bootstrap=refresh_bootstrap,
    )

    fig = plt.figure(figsize=(17.5, 10.5), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        5,
        width_ratios=[1.0, 1.0, 0.032, 0.18, 0.92],
        height_ratios=[1.0, 1.0],
        wspace=0.38,
        hspace=0.42,
        left=0.07,
        right=0.97,
        top=0.94,
        bottom=0.08,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_cbar = fig.add_subplot(gs[:, 2])
    ax_e = fig.add_subplot(gs[0, 4])
    ax_f = fig.add_subplot(gs[1, 4])

    draw_fel_panel(
        ax_a,
        wt_to_cln_summary,
        probability_column="reference_probability",
        title="MD Wild-Type",
        temperature_kelvin=temperature_kelvin,
        folded=folded,
        fel_max_energy_kcal_mol=fel_max_energy_kcal_mol,
        panel_label="A",
    )
    draw_fel_panel(
        ax_b,
        cln_to_wt_summary,
        probability_column="reference_probability",
        title="MD CLN025",
        temperature_kelvin=temperature_kelvin,
        folded=folded,
        fel_max_energy_kcal_mol=fel_max_energy_kcal_mol,
        panel_label="B",
    )
    draw_fel_panel(
        ax_c,
        cln_to_wt_summary,
        probability_column="target_probability",
        title="FELP Wild-Type (CLN025→WT)",
        temperature_kelvin=temperature_kelvin,
        folded=folded,
        fel_max_energy_kcal_mol=fel_max_energy_kcal_mol,
        panel_label="C",
    )
    draw_fel_panel(
        ax_d,
        wt_to_cln_summary,
        probability_column="target_probability",
        title="FELP CLN025 (WT→CLN025)",
        temperature_kelvin=temperature_kelvin,
        folded=folded,
        fel_max_energy_kcal_mol=fel_max_energy_kcal_mol,
        panel_label="D",
    )

    import matplotlib.cm as mpl_cm
    import matplotlib.colors as mcolors
    from felp.plotting.common import _copy_colormap_for_fel_cap

    cmap = _copy_colormap_for_fel_cap(FEL_COLORMAP)
    norm = mcolors.Normalize(vmin=0.0, vmax=fel_max_energy_kcal_mol, clip=False)
    mappable = mpl_cm.ScalarMappable(norm=norm, cmap=cmap)

    cbar = fig.colorbar(
        mappable,
        cax=ax_cbar,
        orientation="vertical",
        extend="max",
    )
    cbar.set_label(r"Relative free energy (kcal mol$^{-1}$)", fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    method_order = ["experiment", "md", "felp"]
    method_labels = ["Experiment", "MD", "FELP"]
    x_groups = np.arange(len(method_order), dtype=float)
    bar_width = 0.34

    def _plot_errorbar(ax, x: float, y: float, entry: dict[str, float]) -> None:
        if "ci_low" in entry:
            ci_low = entry["ci_low"]
            ci_high = entry["ci_high"]
        else:
            ci_low = entry["delta_g_ci_low"]
            ci_high = entry["delta_g_ci_high"]
        if ci_high - ci_low <= 1.0e-8:
            return
        ax.errorbar(
            x,
            y,
            yerr=[[y - ci_low], [ci_high - y]],
            fmt="none",
            ecolor="#222222",
            elinewidth=1.2,
            capsize=4.5,
            capthick=1.2,
            zorder=4,
        )

    panel_f_ylabel = (
        r"$\Delta\Delta G_{\mathrm{fold}}$" + "\n"
        + r"(CLN025 $-$ WT, kcal mol$^{-1}$)"
    )

    for panel_label, ax, ylabel, plot_mode in (
        ("E", ax_e, r"$\Delta G_{\mathrm{fold}}$ (kcal mol$^{-1}$)", "delta_g"),
        ("F", ax_f, panel_f_ylabel, "delta_delta_g"),
    ):
        ax.text(
            -0.12,
            1.08,
            panel_label,
            transform=ax.transAxes,
            fontsize=24,
            fontweight="bold",
            va="bottom",
            ha="right",
        )
        ax.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)

        if plot_mode == "delta_g":
            for variant_index, (variant, color) in enumerate(PANEL_E_VARIANT_COLORS.items()):
                offset = (variant_index - 0.5) * bar_width
                for group_index, method in enumerate(method_order):
                    entry = thermo[method][variant]
                    y = entry["delta_g"]
                    x = x_groups[group_index] + offset
                    ax.bar(
                        x,
                        y,
                        width=bar_width * 0.92,
                        color=color,
                        edgecolor=BAR_EDGE_COLOR,
                        linewidth=BAR_EDGE_WIDTH,
                        label=variant if group_index == 0 else None,
                        zorder=3,
                    )
                    if method != "experiment":
                        _plot_errorbar(ax, x, y, entry)
            ax.axhline(0.0, color="black", linewidth=0.8, zorder=1)
            ax.set_xticks(x_groups, method_labels)
            ax.set_ylabel(ylabel)
            ax.set_ylim(-2.5, 2.5)
            ax.legend(
                frameon=True,
                framealpha=0.85,
                facecolor="white",
                edgecolor="none",
                loc="upper right",
                title="Variant",
            )
        else:
            for group_index, method in enumerate(method_order):
                entry = thermo["delta_delta_g"][method]
                y = entry["mean"]
                x = x_groups[group_index]
                ax.bar(
                    x,
                    y,
                    width=0.55,
                    color=PANEL_F_BAR_COLOR,
                    edgecolor=BAR_EDGE_COLOR,
                    linewidth=BAR_EDGE_WIDTH,
                    zorder=3,
                )
                if method != "experiment":
                    _plot_errorbar(ax, x, y, entry)
            ax.axhline(0.0, color="black", linewidth=0.8, zorder=1)
            ax.set_xticks(x_groups, method_labels)
            ax.set_ylabel(ylabel)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300)
    fig.savefig(os.path.splitext(output_path)[0] + ".pdf")
    plt.close(fig)

    summary = {
        "temperature_kelvin": temperature_kelvin,
        "bootstrap_samples": n_bootstrap,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_cache_path": cache_path,
        "folded_basin": _folded_basin_dict(folded),
        "thermodynamics": thermo,
        "inputs": {
            "wt_to_cln_summary": wt_to_cln_summary,
            "cln_to_wt_summary": cln_to_wt_summary,
            "wt_to_cln_frame_results": wt_to_cln_frame_results,
            "cln_to_wt_frame_results": cln_to_wt_frame_results,
            "wt_md_distances": wt_md_distances,
            "cln_md_distances": cln_md_distances,
        },
    }
    json_path = os.path.splitext(output_path)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    reweight = "reweight_repartition_exclude_tau100/reweighting_neighbor_pool"
    reweight_dir = "reweight_repartition_exclude_tau100"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="/home/brownbp1/Projects/FELP/chignolin_wterm_wild-type/analysis/folding_landscape_figure/chignolin_folding_landscape_figure.png",
    )
    parser.add_argument("--temperature-kelvin", type=float, default=310.0)
    parser.add_argument(
        "--wt-to-cln-summary",
        default=(
            "/home/brownbp1/Projects/FELP/chignolin_wterm_wild-type/"
            f"to_1Y_10Y.redo-1/{reweight}/microstate_summary.csv"
        ),
    )
    parser.add_argument(
        "--cln-to-wt-summary",
        default=(
            "/home/brownbp1/Projects/FELP/chignolin_wterm_CLN025/"
            f"to_1G_10G.redo-1/{reweight}/microstate_summary.csv"
        ),
    )
    parser.add_argument(
        "--wt-to-cln-frame-results",
        default=(
            "/home/brownbp1/Projects/FELP/chignolin_wterm_wild-type/"
            f"to_1Y_10Y.redo-1/{reweight_dir}/frame_results.csv"
        ),
    )
    parser.add_argument(
        "--cln-to-wt-frame-results",
        default=(
            "/home/brownbp1/Projects/FELP/chignolin_wterm_CLN025/"
            f"to_1G_10G.redo-1/{reweight_dir}/frame_results.csv"
        ),
    )
    parser.add_argument(
        "--wt-md-distances",
        default="/home/brownbp1/Projects/FELP/chignolin_wterm_wild-type/distances.dat",
    )
    parser.add_argument(
        "--cln-md-distances",
        default="/home/brownbp1/Projects/FELP/chignolin_wterm_CLN025/distances.dat",
    )
    parser.add_argument("--fel-max-energy-kcal-mol", type=float, default=DEFAULT_FEL_MAX_KCAL_MOL)
    parser.add_argument("--folded-ends-min", type=float, default=RECOMMENDED_FOLDED.ends_min)
    parser.add_argument("--folded-ends-max", type=float, default=RECOMMENDED_FOLDED.ends_max)
    parser.add_argument("--folded-turn-min", type=float, default=RECOMMENDED_FOLDED.turn_min)
    parser.add_argument("--folded-turn-max", type=float, default=RECOMMENDED_FOLDED.turn_max)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--bootstrap-cache",
        default=None,
        help="Path to bootstrap cache JSON (default: <output>_bootstrap_cache.json).",
    )
    parser.add_argument(
        "--refresh-bootstrap",
        action="store_true",
        help="Ignore cached bootstrap results and recompute.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folded = BasinSpec(
        args.folded_ends_min,
        args.folded_ends_max,
        args.folded_turn_min,
        args.folded_turn_max,
    )
    summary = plot_figure(
        args.output,
        folded=folded,
        temperature_kelvin=args.temperature_kelvin,
        wt_to_cln_summary=args.wt_to_cln_summary,
        cln_to_wt_summary=args.cln_to_wt_summary,
        wt_to_cln_frame_results=args.wt_to_cln_frame_results,
        cln_to_wt_frame_results=args.cln_to_wt_frame_results,
        wt_md_distances=args.wt_md_distances,
        cln_md_distances=args.cln_md_distances,
        fel_max_energy_kcal_mol=args.fel_max_energy_kcal_mol,
        n_bootstrap=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_cache_path=args.bootstrap_cache,
        refresh_bootstrap=args.refresh_bootstrap,
    )
    print(json.dumps(summary["thermodynamics"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
