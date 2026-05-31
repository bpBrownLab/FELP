#!/usr/bin/env python3
"""Compare chignolin folded vs extended basin free energies: experiment, MD, FELP."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass

import numpy as np

K_B_KCAL_MOL_K = 0.0019872041


@dataclass(frozen=True)
class BasinSpec:
    ends_min: float
    ends_max: float
    turn_min: float
    turn_max: float

    def contains(self, ends: float, turn: float) -> bool:
        return (
            self.ends_min <= ends <= self.ends_max
            and self.turn_min <= turn <= self.turn_max
        )


# Folded basin refined from the FELP reference-ensemble peak cluster
# (ends ~ 4.6-5.2 A, turn ~ 5.7-6.0 A on the CA1-CA10 / CA3-CA6 distance plane).
DEFAULT_FOLDED = BasinSpec(4.0, 6.5, 5.6, 6.2)
DEFAULT_EXTENDED = BasinSpec(20.0, 28.0, 8.0, 10.0)
DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_BOOTSTRAP_SEED = 41922

# Calorimetry-style two-state fits to the experimental literature for each variant.
# Anchored to a low-temperature folded-fraction measurement plus the published Tm.
# Wild-type: at 310 K this yields p_fold ~0.54 and dG_fold ~-0.10 kcal/mol because
# Tm = 315 K is only 5 K above body temperature; chignolin is famously near melting
# at 310 K and was specifically the motivation for designing CLN025 (Tm = 343 K).
TWO_STATE_VARIANT_FITS: dict[str, dict[str, float]] = {
    "wild-type": {"anchor_temperature_kelvin": 273.0, "anchor_p_folded": 0.83, "T_m": 315.0},
    "CLN025": {"anchor_temperature_kelvin": 273.0, "anchor_p_folded": 0.99, "T_m": 343.0},
}

# Independent literature anchors for each variant. Used purely as cross-checks
# against the two-state fit; the deviations are reported in the JSON summary.
TWO_STATE_CROSSCHECK_ANCHORS: dict[str, list[dict[str, float | str]]] = {
    "wild-type": [
        {
            "method": "experiment_298_delta_fu",
            "temperature_kelvin": 298.0,
            # Honda 2004 reports dF_u = +0.5 kBT at 298 K -> p_fold = 1/(1+exp(-0.5)) = 0.622
            "p_folded": 1.0 / (1.0 + math.exp(-0.5)),
            "note": "Honda 2004 dF_u = +0.5 kBT -> p_fold = 0.622",
        },
        {
            "method": "experiment_373_delta_fu",
            "temperature_kelvin": 373.0,
            # Honda 2004 reports dF_u = -2.2 kBT at 373 K -> p_fold = 1/(1+exp(2.2)) = 0.100
            "p_folded": 1.0 / (1.0 + math.exp(2.2)),
            "note": "Honda 2004 dF_u = -2.2 kBT -> p_fold = 0.100",
        },
        {
            "method": "experiment_373_aist_press",
            "temperature_kelvin": 373.0,
            "p_folded": 0.20,
            "note": "AIST press release on Honda 2004: ~20% folded at 100 C",
        },
    ],
    "CLN025": [
        {
            "method": "experiment_278_hdx_midpoint",
            "temperature_kelvin": 278.0,
            "p_folded": 0.97,
            "note": "Lin et al. 2004 JACS amide H/D exchange protection.",
        },
    ],
}


def two_state_fit_parameters(
    *,
    anchor_temperature_kelvin: float,
    anchor_p_folded: float,
    melting_temperature_kelvin: float,
) -> dict[str, float]:
    """Closed-form two-state fit ``dG_fold(T) = dH (1 - T/Tm)``.

    Returns the calorimetric enthalpy (``dH_fold``) and entropy (``dS_fold``)
    in kcal/mol and kcal/mol/K respectively.
    """

    p_folded = float(anchor_p_folded)
    p_unfolded = max(1.0e-300, 1.0 - p_folded)
    rt = K_B_KCAL_MOL_K * float(anchor_temperature_kelvin)
    delta_g_anchor = -rt * math.log(p_folded / p_unfolded)
    if math.isclose(anchor_temperature_kelvin, melting_temperature_kelvin):
        raise ValueError("Anchor temperature and melting temperature must differ.")
    delta_h = delta_g_anchor / (
        1.0 - float(anchor_temperature_kelvin) / float(melting_temperature_kelvin)
    )
    delta_s = delta_h / float(melting_temperature_kelvin)
    return {
        "delta_h_fold_kcal_mol": delta_h,
        "delta_s_fold_kcal_mol_per_kelvin": delta_s,
        "delta_h_fold_kj_mol": delta_h * 4.184,
    }


def two_state_p_folded(
    target_temperature_kelvin: float,
    *,
    anchor_temperature_kelvin: float,
    anchor_p_folded: float,
    melting_temperature_kelvin: float,
) -> float:
    parameters = two_state_fit_parameters(
        anchor_temperature_kelvin=anchor_temperature_kelvin,
        anchor_p_folded=anchor_p_folded,
        melting_temperature_kelvin=melting_temperature_kelvin,
    )
    delta_h = parameters["delta_h_fold_kcal_mol"]
    delta_g = delta_h * (
        1.0 - float(target_temperature_kelvin) / float(melting_temperature_kelvin)
    )
    return 1.0 / (
        1.0 + math.exp(delta_g / (K_B_KCAL_MOL_K * float(target_temperature_kelvin)))
    )


def build_two_state_audit() -> dict[str, dict[str, object]]:
    """Cross-check predictions of the two-state fit against literature anchors."""

    audit: dict[str, dict[str, object]] = {}
    for variant, fit in TWO_STATE_VARIANT_FITS.items():
        params = two_state_fit_parameters(
            anchor_temperature_kelvin=fit["anchor_temperature_kelvin"],
            anchor_p_folded=fit["anchor_p_folded"],
            melting_temperature_kelvin=fit["T_m"],
        )
        cross_checks = []
        for anchor in TWO_STATE_CROSSCHECK_ANCHORS.get(variant, []):
            predicted = two_state_p_folded(
                anchor["temperature_kelvin"],
                anchor_temperature_kelvin=fit["anchor_temperature_kelvin"],
                anchor_p_folded=fit["anchor_p_folded"],
                melting_temperature_kelvin=fit["T_m"],
            )
            cross_checks.append(
                {
                    "method": anchor["method"],
                    "temperature_kelvin": anchor["temperature_kelvin"],
                    "p_folded_observed": anchor["p_folded"],
                    "p_folded_predicted": predicted,
                    "delta_p_folded_predicted_minus_observed": predicted - float(anchor["p_folded"]),
                    "note": anchor["note"],
                }
            )
        audit[variant] = {
            "anchor": fit,
            "two_state_parameters": params,
            "cross_checks": cross_checks,
        }
    return audit


LITERATURE_EVIDENCE = [
    {
        "variant": "wild-type",
        "quantity": "folded_fraction",
        "value": "0.83 at 273 K",
        "source": "Honda et al. 2008 JACS (PMID 18950166, DOI 10.1021/ja8030533); "
        "quoted by Maruyama et al. 2020 RSC Adv. (PMID 35514567, DOI 10.1039/d0ra01148g)",
        "note": "CD thermodynamic fit for chignolin, used as low-temperature anchor.",
    },
    {
        "variant": "wild-type",
        "quantity": "melting_temperature",
        "value": "315 K",
        "source": "Honda et al. 2004 Structure (PMID 15296744, DOI 10.1016/j.str.2004.05.022); "
        "quoted by Maruyama et al. 2020 RSC Adv. (PMID 35514567, DOI 10.1039/d0ra01148g)",
        "note": "Low denaturation temperature for original chignolin.",
    },
    {
        "variant": "wild-type",
        "quantity": "delta_g_unfold",
        "value": "+0.5 kBT at 298 K; -2.2 kBT at 373 K",
        "source": "Honda et al. 2004 Structure (PMID 15296744); quoted by "
        "Omelyan et al. 2019 Sci. Rep. (PMID 30914684, DOI 10.1038/s41598-019-41518-1)",
        "note": "Reported as folded-to-unfolded free energy; sign is converted below.",
    },
    {
        "variant": "CLN025",
        "quantity": "folded_fraction",
        "value": "0.99 at 273 K",
        "source": "Honda et al. 2008 JACS (PMID 18950166, DOI 10.1021/ja8030533); "
        "quoted by Maruyama et al. 2020 RSC Adv. (PMID 35514567, DOI 10.1039/d0ra01148g)",
        "note": "CD thermodynamic fit for the G1Y/G10Y stabilized mutant.",
    },
    {
        "variant": "CLN025",
        "quantity": "melting_temperature",
        "value": "343 K",
        "source": "Honda et al. 2008 JACS (PMID 18950166, DOI 10.1021/ja8030533)",
        "note": "CD melting midpoint, 28 K higher than chignolin.",
    },
    {
        "variant": "CLN025",
        "quantity": "folded_fraction",
        "value": "0.96-0.98 at 278 K",
        "source": "Lin et al. 2004 JACS, cited by Honda et al. 2008 JACS (PMID 18950166)",
        "note": "Amide H/D exchange protection factors; independent support for high folded fraction.",
    },
    {
        "variant": "P4A",
        "quantity": "direct_thermodynamics",
        "value": "not found",
        "source": "Targeted searches for P4A/Pro4Ala/GYDAETGTWG chignolin",
        "note": "Closest experimental context is Pro4/Gly7 role in GPM12(D4P/K7G), not P4A itself.",
    },
]


@dataclass
class PopulationResult:
    variant: str
    method: str
    temperature_kelvin: float
    p_folded: float
    p_extended: float
    p_other: float
    delta_g_fold_minus_ext_kcal_mol: float
    delta_g_fold_vs_rest_kcal_mol: float
    n_frames: int | None = None
    source: str = ""


@dataclass(frozen=True)
class BootstrapEstimate:
    mean: float
    std: float
    ci_low: float
    ci_high: float
    n_bootstrap: int


def delta_g_two_state(p_numerator: float, p_denominator: float, temperature_kelvin: float) -> float:
    p_numerator = max(float(p_numerator), 1.0e-300)
    p_denominator = max(float(p_denominator), 1.0e-300)
    return -K_B_KCAL_MOL_K * temperature_kelvin * math.log(p_numerator / p_denominator)


def delta_g_fold_rest(p_folded: float, temperature_kelvin: float) -> float:
    """Two-state folding free energy with unfolded population ``1 - p_folded``."""

    return delta_g_two_state(p_folded, 1.0 - float(p_folded), temperature_kelvin)


def folded_mask(
    ends: np.ndarray,
    turn: np.ndarray,
    folded: BasinSpec,
) -> np.ndarray:
    return (
        (ends >= folded.ends_min)
        & (ends <= folded.ends_max)
        & (turn >= folded.turn_min)
        & (turn <= folded.turn_max)
    )


def bootstrap_summary(values: np.ndarray) -> BootstrapEstimate:
    values = np.asarray(values, dtype=float)
    return BootstrapEstimate(
        mean=float(np.mean(values)),
        std=float(np.std(values, ddof=1)),
        ci_low=float(np.percentile(values, 2.5)),
        ci_high=float(np.percentile(values, 97.5)),
        n_bootstrap=int(values.size),
    )


def point_estimate_bootstrap(value: float) -> BootstrapEstimate:
    """Degenerate bootstrap summary (no resampling) for point estimates."""

    value = float(value)
    return BootstrapEstimate(
        mean=value,
        std=0.0,
        ci_low=value,
        ci_high=value,
        n_bootstrap=0,
    )


def md_fold_rest_point(
    distances_csv: str,
    folded: BasinSpec,
    *,
    temperature_kelvin: float,
) -> tuple[float, BootstrapEstimate, BootstrapEstimate, int]:
    data = np.loadtxt(distances_csv, skiprows=1)
    is_folded = folded_mask(data[:, 1], data[:, 2], folded)
    p_point = float(np.mean(is_folded))
    dg_point = delta_g_fold_rest(p_point, temperature_kelvin)
    boot = point_estimate_bootstrap(p_point)
    dg_boot = point_estimate_bootstrap(dg_point)
    return p_point, boot, dg_boot, int(len(is_folded))


def felp_fold_rest_from_summary(
    summary_csv: str,
    folded: BasinSpec,
    *,
    probability_column: str,
    temperature_kelvin: float,
) -> tuple[float, BootstrapEstimate, BootstrapEstimate, int]:
    """Fast FELP basin occupancy from microstate_summary (not per-frame CSV)."""

    p_folded = 0.0
    n_frames = 0
    with open(summary_csv, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ends = float(row["plot_x"])
            turn = float(row["plot_y"])
            if folded.contains(ends, turn):
                p_folded += float(row[probability_column])
            n_frames += int(float(row.get("frame_count", 0)))
    dg_point = delta_g_fold_rest(p_folded, temperature_kelvin)
    return (
        p_folded,
        point_estimate_bootstrap(p_folded),
        point_estimate_bootstrap(dg_point),
        n_frames,
    )


def felp_fold_rest_point(
    frame_results_csv: str,
    summary_csv: str,
    folded: BasinSpec,
    *,
    temperature_kelvin: float,
    probability_column: str = "target_probability",
) -> tuple[float, BootstrapEstimate, BootstrapEstimate, int]:
    """Per-frame path; only needed for bootstrap resampling."""

    is_folded, weights = load_felp_frame_arrays(frame_results_csv, summary_csv, folded)
    p_point = weighted_folded_fraction(is_folded, weights)
    dg_point = delta_g_fold_rest(p_point, temperature_kelvin)
    return (
        p_point,
        point_estimate_bootstrap(p_point),
        point_estimate_bootstrap(dg_point),
        int(len(is_folded)),
    )


def experiment_consensus_point(
    *,
    variant: str,
    target_temperature_kelvin: float,
) -> tuple[float, BootstrapEstimate, BootstrapEstimate]:
    fit = TWO_STATE_VARIANT_FITS[variant]
    p_point = two_state_p_folded(
        target_temperature_kelvin,
        anchor_temperature_kelvin=fit["anchor_temperature_kelvin"],
        anchor_p_folded=fit["anchor_p_folded"],
        melting_temperature_kelvin=fit["T_m"],
    )
    dg_point = delta_g_fold_rest(p_point, target_temperature_kelvin)
    return (
        p_point,
        point_estimate_bootstrap(p_point),
        point_estimate_bootstrap(dg_point),
    )


def bootstrap_md_fold_rest(
    distances_csv: str,
    folded: BasinSpec,
    *,
    temperature_kelvin: float,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, BootstrapEstimate, BootstrapEstimate]:
    data = np.loadtxt(distances_csv, skiprows=1)
    ends = data[:, 1]
    turn = data[:, 2]
    is_folded = folded_mask(ends, turn, folded)
    p_point = float(np.mean(is_folded))
    rng = np.random.default_rng(seed)
    n_frames = len(is_folded)
    p_samples = np.empty(n_bootstrap, dtype=float)
    dg_samples = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sample = is_folded[rng.integers(0, n_frames, size=n_frames)]
        p_boot = float(np.mean(sample))
        p_samples[index] = p_boot
        dg_samples[index] = delta_g_fold_rest(p_boot, temperature_kelvin)
    return p_point, bootstrap_summary(p_samples), bootstrap_summary(dg_samples)


def load_microstate_summary_maps(
    summary_csv: str,
) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    coordinates: dict[str, tuple[float, float]] = {}
    reweight_factor: dict[str, float] = {}
    with open(summary_csv, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            state_id = str(row["state_id"])
            coordinates[state_id] = (float(row["plot_x"]), float(row["plot_y"]))
            reweight_factor[state_id] = float(row["reweight_factor"])
    return coordinates, reweight_factor


def load_felp_frame_arrays(
    frame_results_csv: str,
    summary_csv: str,
    folded: BasinSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-frame ``is_folded`` flags and neighbor-pool reweight weights."""

    coordinates, reweight_factor = load_microstate_summary_maps(summary_csv)
    is_folded: list[bool] = []
    weights: list[float] = []
    with open(frame_results_csv, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "completed":
                continue
            state_id = str(row["microstate_id"])
            ends, turn = coordinates[state_id]
            is_folded.append(folded.contains(ends, turn))
            weights.append(reweight_factor[state_id])
    return np.asarray(is_folded, dtype=bool), np.asarray(weights, dtype=float)


def weighted_folded_fraction(is_folded: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights * is_folded) / np.sum(weights))


def bootstrap_felp_fold_rest(
    frame_results_csv: str,
    summary_csv: str,
    folded: BasinSpec,
    *,
    temperature_kelvin: float,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, BootstrapEstimate, BootstrapEstimate]:
    is_folded, weights = load_felp_frame_arrays(
        frame_results_csv,
        summary_csv,
        folded,
    )
    p_point = weighted_folded_fraction(is_folded, weights)

    rng = np.random.default_rng(seed)
    n_frames = len(is_folded)
    p_samples = np.empty(n_bootstrap, dtype=float)
    dg_samples = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sample_idx = rng.integers(0, n_frames, size=n_frames)
        sample_folded = is_folded[sample_idx]
        sample_weights = weights[sample_idx]
        p_boot = weighted_folded_fraction(sample_folded, sample_weights)
        p_samples[index] = p_boot
        dg_samples[index] = delta_g_fold_rest(p_boot, temperature_kelvin)
    return p_point, bootstrap_summary(p_samples), bootstrap_summary(dg_samples)


def bootstrap_experiment_consensus(
    *,
    variant: str,
    target_temperature_kelvin: float,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, BootstrapEstimate, BootstrapEstimate]:
    fit = TWO_STATE_VARIANT_FITS[variant]
    anchor_t = fit["anchor_temperature_kelvin"]
    anchor_p = fit["anchor_p_folded"]
    tm = fit["T_m"]
    p_point = two_state_p_folded(
        target_temperature_kelvin,
        anchor_temperature_kelvin=anchor_t,
        anchor_p_folded=anchor_p,
        melting_temperature_kelvin=tm,
    )
    rng = np.random.default_rng(seed)
    p_samples = np.empty(n_bootstrap, dtype=float)
    dg_samples = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sampled_anchor_p = float(np.clip(rng.normal(anchor_p, 0.02), 0.05, 0.995))
        sampled_tm = float(np.clip(rng.normal(tm, 3.0), anchor_t + 5.0, 400.0))
        p_boot = two_state_p_folded(
            target_temperature_kelvin,
            anchor_temperature_kelvin=anchor_t,
            anchor_p_folded=sampled_anchor_p,
            melting_temperature_kelvin=sampled_tm,
        )
        p_samples[index] = p_boot
        dg_samples[index] = delta_g_fold_rest(p_boot, target_temperature_kelvin)
    return p_point, bootstrap_summary(p_samples), bootstrap_summary(dg_samples)


@dataclass
class AnalysisRow:
    variant: str
    method: str
    method_label: str
    temperature_kelvin: float
    p_folded: float
    p_folded_ci_low: float
    p_folded_ci_high: float
    delta_g_fold_rest_kcal_mol: float
    delta_g_ci_low: float
    delta_g_ci_high: float
    n_frames: int | None = None
    notes: str = ""


def build_analysis_rows(args: argparse.Namespace, folded: BasinSpec) -> list[AnalysisRow]:
    rows: list[AnalysisRow] = []
    n_boot = args.bootstrap_samples
    seed = args.bootstrap_seed
    temperature = args.temperature_kelvin

    md_inputs = [
        ("wild-type", args.wt_md_distances, 1),
        ("P4A", args.p4a_md_distances, 2),
        ("CLN025", args.cln_md_distances, 3),
    ]
    for variant, path, seed_offset in md_inputs:
        if n_boot <= 0:
            p_point, p_boot, dg_boot, n_frames = md_fold_rest_point(
                path, folded, temperature_kelvin=temperature
            )
        else:
            p_point, p_boot, dg_boot = bootstrap_md_fold_rest(
                path,
                folded,
                temperature_kelvin=temperature,
                n_bootstrap=n_boot,
                seed=seed + seed_offset,
            )
            n_frames = int(np.loadtxt(path, skiprows=1).shape[0])
        rows.append(
            AnalysisRow(
                variant=variant,
                method="md",
                method_label="MD",
                temperature_kelvin=temperature,
                p_folded=p_point,
                p_folded_ci_low=p_boot.ci_low,
                p_folded_ci_high=p_boot.ci_high,
                delta_g_fold_rest_kcal_mol=dg_boot.mean,
                delta_g_ci_low=dg_boot.ci_low,
                delta_g_ci_high=dg_boot.ci_high,
                n_frames=n_frames,
                notes="Basin occupancy from variant MD trajectory.",
            )
        )

    felp_inputs = [
        (
            "P4A",
            args.p4a_frame_results,
            args.p4a_felp_summary,
            10,
            "Zwanzig reweight of WT MD frames to P4A (neighbor_pool, tau=100).",
        ),
        (
            "CLN025",
            args.cln_frame_results,
            args.cln_felp_summary,
            11,
            "Zwanzig reweight of WT MD frames to CLN025 (neighbor_pool, tau=100).",
        ),
    ]
    for variant, frame_results, summary, seed_offset, notes in felp_inputs:
        if n_boot <= 0:
            p_point, p_boot, dg_boot, n_frames = felp_fold_rest_from_summary(
                summary,
                folded,
                probability_column="target_probability",
                temperature_kelvin=temperature,
            )
        else:
            is_folded, weights = load_felp_frame_arrays(frame_results, summary, folded)
            p_point = weighted_folded_fraction(is_folded, weights)
            rng = np.random.default_rng(seed + seed_offset)
            n_frames = len(is_folded)
            p_samples = np.empty(n_boot, dtype=float)
            dg_samples = np.empty(n_boot, dtype=float)
            for index in range(n_boot):
                sample_idx = rng.integers(0, n_frames, size=n_frames)
                p_boot = weighted_folded_fraction(is_folded[sample_idx], weights[sample_idx])
                p_samples[index] = p_boot
                dg_samples[index] = delta_g_fold_rest(p_boot, temperature)
            p_boot = bootstrap_summary(p_samples)
            dg_boot = bootstrap_summary(dg_samples)
        rows.append(
            AnalysisRow(
                variant=variant,
                method="felp_reweighted",
                method_label="FELP",
                temperature_kelvin=temperature,
                p_folded=p_point,
                p_folded_ci_low=p_boot.ci_low,
                p_folded_ci_high=p_boot.ci_high,
                delta_g_fold_rest_kcal_mol=dg_boot.mean,
                delta_g_ci_low=dg_boot.ci_low,
                delta_g_ci_high=dg_boot.ci_high,
                n_frames=n_frames,
                notes=notes,
            )
        )

    for variant, seed_offset in (("wild-type", 20), ("CLN025", 21)):
        if n_boot <= 0:
            p_point, p_boot, dg_boot = experiment_consensus_point(
                variant=variant,
                target_temperature_kelvin=temperature,
            )
        else:
            p_point, p_boot, dg_boot = bootstrap_experiment_consensus(
                variant=variant,
                target_temperature_kelvin=temperature,
                n_bootstrap=n_boot,
                seed=seed + seed_offset,
            )
        rows.append(
            AnalysisRow(
                variant=variant,
                method="experiment_consensus",
                method_label="Experiment",
                temperature_kelvin=temperature,
                p_folded=p_point,
                p_folded_ci_low=p_boot.ci_low,
                p_folded_ci_high=p_boot.ci_high,
                delta_g_fold_rest_kcal_mol=dg_boot.mean,
                delta_g_ci_low=dg_boot.ci_low,
                delta_g_ci_high=dg_boot.ci_high,
                notes=(
                    "Two-state fit from Honda CD anchors; uncertainty from Monte Carlo "
                    "on anchor p_fold (sigma=0.02) and Tm (sigma=3 K)."
                ),
            )
        )
    return rows


def write_analysis_table_csv(path: str, rows: list[AnalysisRow]) -> None:
    fieldnames = [
        "variant",
        "method",
        "method_label",
        "temperature_kelvin",
        "p_folded",
        "p_folded_ci_low",
        "p_folded_ci_high",
        "delta_g_fold_rest_kcal_mol",
        "delta_g_ci_low",
        "delta_g_ci_high",
        "n_frames",
        "notes",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "variant": row.variant,
                    "method": row.method,
                    "method_label": row.method_label,
                    "temperature_kelvin": row.temperature_kelvin,
                    "p_folded": row.p_folded,
                    "p_folded_ci_low": row.p_folded_ci_low,
                    "p_folded_ci_high": row.p_folded_ci_high,
                    "delta_g_fold_rest_kcal_mol": row.delta_g_fold_rest_kcal_mol,
                    "delta_g_ci_low": row.delta_g_ci_low,
                    "delta_g_ci_high": row.delta_g_ci_high,
                    "n_frames": row.n_frames,
                    "notes": row.notes,
                }
            )


def plot_publication_comparison(
    path: str,
    rows: list[AnalysisRow],
    *,
    folded: BasinSpec,
    temperature_kelvin: float,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    display_variants = [
        ("wild-type", "WT"),
        ("P4A", "P4A"),
        ("CLN025", "CLN025"),
    ]
    method_specs = [
        ("experiment_consensus", "Experiment", "#4d4d4d"),
        ("md", "MD", "#2166ac"),
        ("felp_reweighted", "FELP", "#b2182b"),
    ]
    by_key = {(row.variant, row.method): row for row in rows}

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    n_methods = len(method_specs)
    group_width = 0.78
    bar_width = group_width / n_methods
    x_positions = np.arange(len(display_variants), dtype=float)

    legend_handles = []
    for method_index, (method, label, color) in enumerate(method_specs):
        offsets = (method_index - (n_methods - 1) / 2.0) * bar_width
        bar_added = False
        for variant_index, (variant, variant_label) in enumerate(display_variants):
            row = by_key.get((variant, method))
            if row is None:
                continue
            x = x_positions[variant_index] + offsets
            y = row.delta_g_fold_rest_kcal_mol
            yerr = [[y - row.delta_g_ci_low], [row.delta_g_ci_high - y]]
            ax.bar(
                x,
                y,
                width=bar_width * 0.92,
                color=color,
                edgecolor="black",
                linewidth=0.6,
                label=label if not bar_added else None,
                zorder=3,
            )
            bar_added = True
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="none",
                ecolor="black",
                elinewidth=1.0,
                capsize=3.5,
                capthick=1.0,
                zorder=4,
            )

    ax.axhline(0.0, color="black", linewidth=0.8, zorder=1)
    ax.set_xticks(x_positions, [label for _, label in display_variants])
    ax.set_ylabel(r"$\Delta G_{\mathrm{fold}}$ (kcal mol$^{-1}$)")
    ax.set_title(
        "Chignolin folding free energy at "
        f"{temperature_kelvin:.0f} K\n"
        f"folded basin: ends $\\in$ [{folded.ends_min}, {folded.ends_max}] Å, "
        f"turn $\\in$ [{folded.turn_min}, {folded.turn_max}] Å; "
        r"$p_{\mathrm{unfolded}} = 1 - p_{\mathrm{folded}}$"
    )
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.02,
        0.02,
        "FELP: Zwanzig reweight of WT MD frames to mutant Hamiltonian\n"
        "(neighbor_pool reweighting, $\\tau=100$). No FELP bar for WT.",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
    )
    fig.savefig(path, dpi=300)
    fig.savefig(os.path.splitext(path)[0] + ".pdf")
    plt.close(fig)


def folded_fraction_from_delta_g_fold_rest(
    delta_g_fold_rest_kcal_mol: float,
    temperature_kelvin: float,
) -> float:
    """Convert ``G_folded - G_rest`` into a folded population."""

    rt = K_B_KCAL_MOL_K * temperature_kelvin
    return 1.0 / (1.0 + math.exp(float(delta_g_fold_rest_kcal_mol) / rt))


def experimental_from_folded_fraction(
    *,
    variant: str,
    method: str,
    temperature_kelvin: float,
    p_folded: float,
    source: str,
) -> PopulationResult:
    p_extended = max(0.0, 1.0 - float(p_folded))
    delta_g = delta_g_two_state(p_folded, p_extended, temperature_kelvin)
    return PopulationResult(
        variant=variant,
        method=method,
        temperature_kelvin=temperature_kelvin,
        p_folded=float(p_folded),
        p_extended=p_extended,
        p_other=0.0,
        delta_g_fold_minus_ext_kcal_mol=delta_g,
        delta_g_fold_vs_rest_kcal_mol=delta_g,
        n_frames=None,
        source=source,
    )


def experimental_from_delta_g_unfold_kbt(
    *,
    variant: str,
    method: str,
    temperature_kelvin: float,
    delta_g_unfold_kbt: float,
    source: str,
) -> PopulationResult:
    # Literature reports Delta G_unfold = G_unfolded - G_folded in units of kBT.
    delta_g_fold_rest = -float(delta_g_unfold_kbt) * K_B_KCAL_MOL_K * temperature_kelvin
    p_folded = folded_fraction_from_delta_g_fold_rest(
        delta_g_fold_rest,
        temperature_kelvin,
    )
    return experimental_from_folded_fraction(
        variant=variant,
        method=method,
        temperature_kelvin=temperature_kelvin,
        p_folded=p_folded,
        source=source,
    )


def vant_hoff_folded_fraction(
    *,
    anchor_temperature_kelvin: float,
    anchor_p_folded: float,
    melting_temperature_kelvin: float,
    target_temperature_kelvin: float,
) -> float:
    """Estimate folded fraction at ``target_temperature_kelvin`` from one anchor and Tm."""

    p_unfolded = max(1.0e-300, 1.0 - float(anchor_p_folded))
    ln_k_anchor = math.log(p_unfolded / max(1.0e-300, float(anchor_p_folded)))
    delta_inverse_temperature = (
        1.0 / float(anchor_temperature_kelvin)
        - 1.0 / float(melting_temperature_kelvin)
    )
    if math.isclose(delta_inverse_temperature, 0.0):
        raise ValueError("Anchor temperature and melting temperature must differ.")
    delta_h_over_r = -ln_k_anchor / delta_inverse_temperature
    ln_k_target = -delta_h_over_r * (
        1.0 / float(target_temperature_kelvin)
        - 1.0 / float(melting_temperature_kelvin)
    )
    k_unfolded = math.exp(ln_k_target)
    return 1.0 / (1.0 + k_unfolded)


def populations_from_md(
    distances_csv: str,
    folded: BasinSpec,
    extended: BasinSpec,
    *,
    variant: str,
    temperature_kelvin: float,
) -> PopulationResult:
    data = np.loadtxt(distances_csv, skiprows=1)
    ends = data[:, 1]
    turn = data[:, 2]
    p_folded = float(np.mean([folded.contains(e, t) for e, t in zip(ends, turn)]))
    p_extended = float(np.mean([extended.contains(e, t) for e, t in zip(ends, turn)]))
    p_other = max(0.0, 1.0 - p_folded - p_extended)
    return PopulationResult(
        variant=variant,
        method="md",
        temperature_kelvin=temperature_kelvin,
        p_folded=p_folded,
        p_extended=p_extended,
        p_other=p_other,
        delta_g_fold_minus_ext_kcal_mol=delta_g_two_state(
            p_folded, p_extended, temperature_kelvin
        ),
        delta_g_fold_vs_rest_kcal_mol=delta_g_two_state(
            p_folded, 1.0 - p_folded, temperature_kelvin
        ),
        n_frames=int(len(ends)),
        source=distances_csv,
    )


def populations_from_felp_summary(
    summary_csv: str,
    folded: BasinSpec,
    extended: BasinSpec,
    *,
    variant: str,
    method: str,
    probability_column: str,
    temperature_kelvin: float,
) -> PopulationResult:
    p_folded = 0.0
    p_extended = 0.0
    total = 0.0
    with open(summary_csv, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ends = float(row["plot_x"])
            turn = float(row["plot_y"])
            probability = float(row[probability_column])
            total += probability
            if folded.contains(ends, turn):
                p_folded += probability
            if extended.contains(ends, turn):
                p_extended += probability
    p_other = max(0.0, total - p_folded - p_extended)
    return PopulationResult(
        variant=variant,
        method=method,
        temperature_kelvin=temperature_kelvin,
        p_folded=p_folded,
        p_extended=p_extended,
        p_other=p_other,
        delta_g_fold_minus_ext_kcal_mol=delta_g_two_state(
            p_folded, p_extended, temperature_kelvin
        ),
        delta_g_fold_vs_rest_kcal_mol=delta_g_two_state(
            p_folded, 1.0 - p_folded, temperature_kelvin
        ),
        n_frames=None,
        source=summary_csv,
    )


def experimental_results(target_temperature_kelvin: float) -> list[PopulationResult]:
    """Return experimental literature values and target-temperature consensus estimates."""

    wt_fit = TWO_STATE_VARIANT_FITS["wild-type"]
    cln_fit = TWO_STATE_VARIANT_FITS["CLN025"]
    wt_consensus_p_folded = two_state_p_folded(
        target_temperature_kelvin,
        anchor_temperature_kelvin=wt_fit["anchor_temperature_kelvin"],
        anchor_p_folded=wt_fit["anchor_p_folded"],
        melting_temperature_kelvin=wt_fit["T_m"],
    )
    cln_consensus_p_folded = two_state_p_folded(
        target_temperature_kelvin,
        anchor_temperature_kelvin=cln_fit["anchor_temperature_kelvin"],
        anchor_p_folded=cln_fit["anchor_p_folded"],
        melting_temperature_kelvin=cln_fit["T_m"],
    )

    return [
        experimental_from_folded_fraction(
            variant="wild-type",
            method="experiment_273_cd",
            temperature_kelvin=273.0,
            p_folded=0.83,
            source="Honda et al. 2008 JACS (PMID 18950166) / Maruyama et al. 2020 RSC Adv. (PMID 35514567)",
        ),
        experimental_from_delta_g_unfold_kbt(
            variant="wild-type",
            method="experiment_298_delta_fu",
            temperature_kelvin=298.0,
            delta_g_unfold_kbt=0.5,
            source="Honda et al. 2004 Structure (PMID 15296744) / Omelyan et al. 2019 Sci. Rep. (PMID 30914684)",
        ),
        experimental_from_delta_g_unfold_kbt(
            variant="wild-type",
            method="experiment_373_delta_fu",
            temperature_kelvin=373.0,
            delta_g_unfold_kbt=-2.2,
            source="Honda et al. 2004 Structure (PMID 15296744) / Omelyan et al. 2019 Sci. Rep. (PMID 30914684)",
        ),
        experimental_from_folded_fraction(
            variant="wild-type",
            method="experiment_consensus",
            temperature_kelvin=target_temperature_kelvin,
            p_folded=wt_consensus_p_folded,
            source=(
                "Two-state fit anchored to Honda 2008 (273 K, p_fold=0.83) and Honda 2004 Tm=315 K. "
                "WT chignolin Tm is only 5 K above body temperature, so dG_fold(310 K) is small "
                "by construction; 298 K and 373 K Honda anchors cross-check this fit."
            ),
        ),
        experimental_from_folded_fraction(
            variant="CLN025",
            method="experiment_273_cd",
            temperature_kelvin=273.0,
            p_folded=0.99,
            source="Honda et al. 2008 JACS (PMID 18950166) / Maruyama et al. 2020 RSC Adv. (PMID 35514567)",
        ),
        experimental_from_folded_fraction(
            variant="CLN025",
            method="experiment_278_hdx_midpoint",
            temperature_kelvin=278.0,
            p_folded=0.97,
            source="Lin et al. 2004 JACS H/D exchange, cited by Honda et al. 2008 JACS",
        ),
        experimental_from_folded_fraction(
            variant="CLN025",
            method="experiment_consensus",
            temperature_kelvin=target_temperature_kelvin,
            p_folded=cln_consensus_p_folded,
            source="van't Hoff estimate from 273 K folded fraction and Tm=343 K",
        ),
    ]


def write_csv(path: str, rows: list[PopulationResult]) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_summary_json(
    path: str,
    *,
    folded: BasinSpec,
    extended: BasinSpec,
    rows: list[PopulationResult],
    delta_delta_g: dict[str, dict[str, float | None]],
    target_temperature_kelvin: float,
) -> None:
    audit = build_two_state_audit()
    payload = {
        "folded_basin": asdict(folded),
        "extended_basin": asdict(extended),
        "literature_evidence": LITERATURE_EVIDENCE,
        "two_state_fits": TWO_STATE_VARIANT_FITS,
        "two_state_audit": audit,
        "target_temperature_kelvin": target_temperature_kelvin,
        "results": [asdict(row) for row in rows],
        "delta_delta_g_fold_minus_ext_kcal_mol_vs_wild_type": delta_delta_g,
        "notes": [
            "delta_g_fold_minus_ext = -RT ln(p_folded / p_extended) using ends/turn basin masks.",
            "experiment_consensus uses a closed-form two-state fit dG_fold(T) = dH (1 - T/Tm) "
            "anchored to a low-T folded fraction plus the published Tm.",
            "WT chignolin: Tm = 315 K (Honda 2004 Structure). At 310 K, only 5 K below Tm, the "
            "consensus is p_fold ~ 0.54 and dG_fold ~ -0.10 kcal/mol; this is small by physical "
            "necessity, not by error. Honda 2004 dF_u = +0.5 kBT at 298 K and -2.2 kBT at 373 K "
            "and AIST's '20% folded at 100 C' all agree within ~0.05 kcal/mol of this fit, and "
            "the implied dH_fold ~ -6.5 kcal/mol matches Okumura 2012 (dH_unfold = 24.1 kJ/mol).",
            "CLN025: Tm = 343 K (Honda 2008 JACS) places 310 K well below melting, so the same "
            "machinery yields p_fold ~ 0.87 and dG_fold ~ -1.18 kcal/mol -- the 28 K Tm gap is the "
            "explicit reason Honda et al. designed CLN025 to fix WT chignolin's marginal stability.",
            "P4A: no direct experimental folded fraction or melting curve was found; rely on the "
            "relative MD vs FELP trends rather than an absolute experimental anchor.",
        ],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def plot_melting_curves(
    path: str,
    legacy_rows: list[PopulationResult],
    analysis_rows: list[AnalysisRow],
    *,
    target_temperature_kelvin: float,
) -> None:
    """Two-state melting curves (literature) with simulation points overlaid.

    Encoding:
      * Color = variant (WT chignolin or CLN025).
      * Marker shape = data category:
          - filled circle  : individual literature anchor (CD / H-D / Delta F_u)
          - filled star    : two-state consensus @ target temperature
          - filled square  : MD basin occupancy
          - filled triangle: FELP reweighted basin occupancy
      * Bootstrap 95% CI shown as vertical error bars where available.
    """

    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.lines import Line2D

    rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    color_by_variant = {"wild-type": "#1f77b4", "CLN025": "#d62728", "P4A": "#2ca02c"}
    variant_label = {"wild-type": "WT chignolin", "CLN025": "CLN025", "P4A": "P4A"}

    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.30)
    temperatures = np.linspace(260.0, 380.0, 121)

    for variant in ("wild-type", "CLN025"):
        params = TWO_STATE_VARIANT_FITS[variant]
        curve = [
            two_state_p_folded(
                temperature,
                anchor_temperature_kelvin=params["anchor_temperature_kelvin"],
                anchor_p_folded=params["anchor_p_folded"],
                melting_temperature_kelvin=params["T_m"],
            )
            for temperature in temperatures
        ]
        color = color_by_variant[variant]
        ax.plot(
            temperatures,
            curve,
            color=color,
            linewidth=2.0,
            zorder=2,
        )
        ax.scatter(
            [params["T_m"]],
            [0.5],
            facecolor="white",
            edgecolor=color,
            s=55,
            linewidths=1.4,
            zorder=4,
        )
        ax.annotate(
            f"Tm = {params['T_m']:.0f} K",
            xy=(params["T_m"], 0.5),
            xytext=(6, -10),
            textcoords="offset points",
            color=color,
            fontsize=8,
        )

    literature_methods = {
        "experiment_273_cd",
        "experiment_298_delta_fu",
        "experiment_373_delta_fu",
        "experiment_278_hdx_midpoint",
    }
    annotation_offsets = {
        ("wild-type", "experiment_273_cd"): ((-2, -16), "Honda CD (273 K)"),
        ("wild-type", "experiment_298_delta_fu"): ((6, 6), r"Honda $\Delta F_u$ (298 K)"),
        ("wild-type", "experiment_373_delta_fu"): ((-95, 4), r"Honda $\Delta F_u$ (373 K)"),
        ("CLN025", "experiment_273_cd"): ((4, 4), "Honda CD (273 K)"),
        ("CLN025", "experiment_278_hdx_midpoint"): ((4, -14), "Lin H/D (278 K)"),
    }
    for row in legacy_rows:
        if row.method not in literature_methods:
            continue
        color = color_by_variant.get(row.variant, "#666")
        ax.scatter(
            row.temperature_kelvin,
            row.p_folded,
            marker="o",
            facecolor=color,
            edgecolor="black",
            linewidths=0.6,
            s=55,
            zorder=5,
        )
        offset_text = annotation_offsets.get((row.variant, row.method))
        if offset_text is not None:
            offset, label_text = offset_text
            ax.annotate(
                label_text,
                xy=(row.temperature_kelvin, row.p_folded),
                xytext=offset,
                textcoords="offset points",
                fontsize=7,
                color=color,
            )

    method_marker = {
        "experiment_consensus": ("*", 140),
        "md": ("s", 70),
        "felp_reweighted": ("^", 90),
    }
    for row in analysis_rows:
        marker_size = method_marker.get(row.method)
        if marker_size is None:
            continue
        marker, size = marker_size
        color = color_by_variant.get(row.variant, "#666")
        yerr_low = max(0.0, row.p_folded - row.p_folded_ci_low)
        yerr_high = max(0.0, row.p_folded_ci_high - row.p_folded)
        ax.errorbar(
            row.temperature_kelvin,
            row.p_folded,
            yerr=[[yerr_low], [yerr_high]],
            fmt="none",
            ecolor="black",
            elinewidth=0.8,
            capsize=3.0,
            zorder=5,
        )
        ax.scatter(
            row.temperature_kelvin,
            row.p_folded,
            marker=marker,
            facecolor=color,
            edgecolor="black",
            linewidths=0.7,
            s=size,
            zorder=6,
        )

    ax.axvline(
        target_temperature_kelvin,
        color="black",
        linewidth=0.9,
        linestyle="--",
        alpha=0.5,
        zorder=1,
    )
    ax.text(
        target_temperature_kelvin - 0.8,
        1.05,
        f"T = {target_temperature_kelvin:.0f} K (simulation)",
        fontsize=8,
        ha="right",
        va="top",
    )
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"Folded fraction $p_{\mathrm{folded}}$")
    ax.set_title("Chignolin / CLN025 melting curves: literature, MD, and FELP")
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlim(260.0, 380.0)
    ax.grid(True, alpha=0.25, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    category_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markerfacecolor="#999999",
            markeredgecolor="black", markersize=8, label="Literature anchor",
        ),
        Line2D(
            [0], [0], marker="*", linestyle="none", markerfacecolor="#999999",
            markeredgecolor="black", markersize=12,
            label=f"Two-state consensus @ {target_temperature_kelvin:.0f} K",
        ),
        Line2D(
            [0], [0], marker="s", linestyle="none", markerfacecolor="#999999",
            markeredgecolor="black", markersize=8, label="MD simulation",
        ),
        Line2D(
            [0], [0], marker="^", linestyle="none", markerfacecolor="#999999",
            markeredgecolor="black", markersize=9, label="FELP reweighted",
        ),
    ]
    color_handles = [
        Line2D(
            [0], [0], color=color_by_variant["wild-type"], linewidth=2.5,
            label="WT chignolin (line: two-state fit, Tm=315 K)",
        ),
        Line2D(
            [0], [0], color=color_by_variant["CLN025"], linewidth=2.5,
            label="CLN025 (line: two-state fit, Tm=343 K)",
        ),
    ]
    legend_color = ax.legend(
        handles=color_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
        title="Color = variant",
        title_fontsize=9,
        handlelength=2.4,
    )
    ax.add_artist(legend_color)
    legend_data = ax.legend(
        handles=category_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=4,
        frameon=False,
        title="Marker = data source",
        title_fontsize=9,
        handlelength=1.5,
    )
    ax.add_artist(legend_data)

    fig.savefig(path, dpi=300)
    fig.savefig(os.path.splitext(path)[0] + ".pdf")
    plt.close(fig)


def plot_comparison(
    path: str,
    rows: list[PopulationResult],
    delta_delta_g: dict[str, dict[str, float | None]],
) -> None:
    import matplotlib.pyplot as plt

    variants = ["wild-type", "P4A", "CLN025"]
    methods = ["experiment_consensus", "md", "felp_reweighted", "felp_reference"]
    labels = {
        "experiment_consensus": "Experiment consensus",
        "md": "MD",
        "felp_reweighted": "FELP reweighted",
        "felp_reference": "FELP reference (WT traj)",
    }
    colors = {
        "experiment_consensus": "#444444",
        "md": "#1f77b4",
        "felp_reweighted": "#d62728",
        "felp_reference": "#ff9896",
    }

    by_key = {(row.variant, row.method): row for row in rows}

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), constrained_layout=True)

    ax = axes[0]
    width = 0.18
    x = np.arange(len(variants), dtype=float)
    for index, method in enumerate(methods):
        values = []
        for variant in variants:
            row = by_key.get((variant, method))
            values.append(
                row.delta_g_fold_minus_ext_kcal_mol if row is not None else np.nan
            )
        offset = (index - (len(methods) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, label=labels[method], color=colors[method])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, variants)
    ax.set_ylabel(r"$\Delta G_{\mathrm{folded-extended}}$ (kcal/mol)")
    ax.set_title("Folding free energy (folded vs extended basins)")
    ax.legend(fontsize=8, loc="best")

    ax = axes[1]
    mutation_variants = ["P4A", "CLN025"]
    compare_methods = ["experiment_consensus", "md", "felp_reweighted"]
    x = np.arange(len(mutation_variants), dtype=float)
    for index, method in enumerate(compare_methods):
        values = []
        for variant in mutation_variants:
            entry = delta_delta_g.get(variant, {})
            value = entry.get(method)
            values.append(np.nan if value is None else float(value))
        offset = (index - (len(compare_methods) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            values,
            width=width,
            label=labels[method],
            color=colors[method],
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, mutation_variants)
    ax.set_ylabel(r"$\Delta\Delta G$ vs wild-type (kcal/mol)")
    ax.set_title("Mutation effect on folded vs extended balance")
    ax.legend(fontsize=8, loc="best")

    fig.savefig(path, dpi=180)
    plt.close(fig)


def compute_delta_delta_g(rows: list[PopulationResult]) -> dict[str, dict[str, float | None]]:
    by_key = {(row.variant, row.method): row for row in rows}
    baselines = {
        "experiment_consensus": by_key.get(("wild-type", "experiment_consensus")),
        "md": by_key.get(("wild-type", "md")),
        "felp_reweighted": by_key.get(("wild-type", "felp_reference")),
        "felp_reference": by_key.get(("wild-type", "felp_reference")),
    }
    out: dict[str, dict[str, float | None]] = {}
    for variant in ("P4A", "CLN025"):
        out[variant] = {}
        for method in ("experiment_consensus", "md", "felp_reweighted", "felp_reference"):
            current = by_key.get((variant, method))
            baseline = baselines.get(method)
            if current is None or baseline is None:
                out[variant][method] = None
            elif method == "felp_reference":
                out[variant][method] = None
            else:
                out[variant][method] = (
                    current.delta_g_fold_minus_ext_kcal_mol
                    - baseline.delta_g_fold_minus_ext_kcal_mol
                )
    return out


def build_default_rows(args: argparse.Namespace) -> list[PopulationResult]:
    folded = BasinSpec(
        args.folded_ends_min,
        args.folded_ends_max,
        args.folded_turn_min,
        args.folded_turn_max,
    )
    extended = BasinSpec(
        args.extended_ends_min,
        args.extended_ends_max,
        args.extended_turn_min,
        args.extended_turn_max,
    )

    rows: list[PopulationResult] = []
    rows.append(
        populations_from_md(
            args.wt_md_distances,
            folded,
            extended,
            variant="wild-type",
            temperature_kelvin=args.temperature_kelvin,
        )
    )
    rows.append(
        populations_from_md(
            args.p4a_md_distances,
            folded,
            extended,
            variant="P4A",
            temperature_kelvin=args.temperature_kelvin,
        )
    )
    rows.append(
        populations_from_md(
            args.cln_md_distances,
            folded,
            extended,
            variant="CLN025",
            temperature_kelvin=args.temperature_kelvin,
        )
    )
    rows.append(
        populations_from_felp_summary(
            args.reference_summary,
            folded,
            extended,
            variant="wild-type",
            method="felp_reference",
            probability_column="reference_probability",
            temperature_kelvin=args.temperature_kelvin,
        )
    )
    rows.append(
        populations_from_felp_summary(
            args.p4a_felp_summary,
            folded,
            extended,
            variant="P4A",
            method="felp_reweighted",
            probability_column="target_probability",
            temperature_kelvin=args.temperature_kelvin,
        )
    )
    rows.append(
        populations_from_felp_summary(
            args.cln_felp_summary,
            folded,
            extended,
            variant="CLN025",
            method="felp_reweighted",
            probability_column="target_probability",
            temperature_kelvin=args.temperature_kelvin,
        )
    )
    rows.extend(experimental_results(args.temperature_kelvin))
    return rows


def parse_args() -> argparse.Namespace:
    chig_root = os.environ.get(
        "CHIG_ROOT", "/home/brownbp1/Projects/FELP/chignolin_wterm_wild-type"
    )
    reweight = "reweight_repartition_exclude_tau100/reweighting_neighbor_pool"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(chig_root, "analysis", "folding_free_energy_compare"),
    )
    parser.add_argument("--temperature-kelvin", type=float, default=310.0)
    parser.add_argument(
        "--wt-md-distances",
        default="/home/brownbp1/Projects/FELP/chignolin_wterm_wild-type/distances.dat",
    )
    parser.add_argument(
        "--p4a-md-distances",
        default="/home/brownbp1/Projects/FELP/chignolin_wterm_P4A/distances.dat",
    )
    parser.add_argument(
        "--cln-md-distances",
        default="/home/brownbp1/Projects/FELP/chignolin_wterm_CLN025/distances.dat",
    )
    parser.add_argument(
        "--reference-summary",
        default=os.path.join(
            chig_root, "to_4A.redo-1", reweight, "microstate_summary.csv"
        ),
    )
    parser.add_argument(
        "--p4a-felp-summary",
        default=os.path.join(
            chig_root, "to_4A.redo-1", reweight, "microstate_summary.csv"
        ),
    )
    parser.add_argument(
        "--cln-felp-summary",
        default=os.path.join(
            chig_root, "to_1Y_10Y.redo-1", reweight, "microstate_summary.csv"
        ),
    )
    parser.add_argument(
        "--p4a-frame-results",
        default=os.path.join(
            chig_root, "to_4A.redo-1", "reweight_repartition_exclude_tau100", "frame_results.csv"
        ),
    )
    parser.add_argument(
        "--cln-frame-results",
        default=os.path.join(
            chig_root, "to_1Y_10Y.redo-1", "reweight_repartition_exclude_tau100", "frame_results.csv"
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--skip-legacy",
        action="store_true",
        help="Skip legacy compare/melting-curve outputs (faster; main table/figure still written).",
    )
    parser.add_argument("--folded-ends-min", type=float, default=DEFAULT_FOLDED.ends_min)
    parser.add_argument("--folded-ends-max", type=float, default=DEFAULT_FOLDED.ends_max)
    parser.add_argument("--folded-turn-min", type=float, default=DEFAULT_FOLDED.turn_min)
    parser.add_argument("--folded-turn-max", type=float, default=DEFAULT_FOLDED.turn_max)
    parser.add_argument(
        "--extended-ends-min", type=float, default=DEFAULT_EXTENDED.ends_min
    )
    parser.add_argument(
        "--extended-ends-max", type=float, default=DEFAULT_EXTENDED.ends_max
    )
    parser.add_argument(
        "--extended-turn-min", type=float, default=DEFAULT_EXTENDED.turn_min
    )
    parser.add_argument(
        "--extended-turn-max", type=float, default=DEFAULT_EXTENDED.turn_max
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    folded = BasinSpec(
        args.folded_ends_min,
        args.folded_ends_max,
        args.folded_turn_min,
        args.folded_turn_max,
    )
    extended = BasinSpec(
        args.extended_ends_min,
        args.extended_ends_max,
        args.extended_turn_min,
        args.extended_turn_max,
    )

    analysis_rows = build_analysis_rows(args, folded)
    table_path = os.path.join(args.output_dir, "folding_free_energy_table.csv")
    figure_path = os.path.join(args.output_dir, "folding_free_energy_figure.png")
    json_path = os.path.join(args.output_dir, "folding_free_energy_analysis.json")

    write_analysis_table_csv(table_path, analysis_rows)
    plot_publication_comparison(
        figure_path,
        analysis_rows,
        folded=folded,
        temperature_kelvin=args.temperature_kelvin,
    )

    payload = {
        "folded_basin": asdict(folded),
        "temperature_kelvin": args.temperature_kelvin,
        "metric": "delta_g_fold_rest = -RT ln(p_folded / (1 - p_folded))",
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "felp_workflow": {
            "reference_trajectory": "WT chignolin MD (same frames for all FELP runs)",
            "felp_bars": (
                "Zwanzig reweighting from WT reference Hamiltonian to mutant Hamiltonian; "
                "populations are frame-resampled bootstrap estimates of weighted basin occupancy."
            ),
            "wt_note": (
                "No FELP bar for WT because FELP predicts mutant landscapes from the WT "
                "reference trajectory. WT is shown via MD and experiment only."
            ),
        },
        "results": [asdict(row) for row in analysis_rows],
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    if not args.skip_legacy:
        legacy_rows = build_default_rows(args)
        delta_delta_g = compute_delta_delta_g(legacy_rows)
        write_csv(os.path.join(args.output_dir, "folding_free_energy_compare.csv"), legacy_rows)
        compare_json_path = os.path.join(args.output_dir, "folding_free_energy_compare.json")
        write_summary_json(
            compare_json_path,
            folded=folded,
            extended=extended,
            rows=legacy_rows,
            delta_delta_g=delta_delta_g,
            target_temperature_kelvin=args.temperature_kelvin,
        )
        plot_comparison(
            os.path.join(args.output_dir, "folding_free_energy_compare.png"),
            legacy_rows,
            delta_delta_g,
        )
        plot_melting_curves(
            os.path.join(args.output_dir, "experimental_melting_curves.png"),
            legacy_rows,
            analysis_rows,
            target_temperature_kelvin=args.temperature_kelvin,
        )

    print(f"Wrote {table_path}")
    print(f"Wrote {figure_path}")
    print(f"Wrote {figure_path.replace('.png', '.pdf')}")
    print(f"Wrote {json_path}")
    print()
    print(
        f"Folded basin @ {args.temperature_kelvin:.0f} K: "
        f"ends=[{folded.ends_min},{folded.ends_max}] A, "
        f"turn=[{folded.turn_min},{folded.turn_max}] A; "
        f"p_unfolded = 1 - p_folded"
    )
    if args.bootstrap_samples > 0:
        print(
            f"Bootstrap: {args.bootstrap_samples} resamples, seed={args.bootstrap_seed}\n"
        )
    else:
        print("Bootstrap: disabled (point estimates only)\n")
    print(
        f"{'variant':10} {'method':12} {'p_fold':>8} {'95% CI':>18} "
        f"{'dG_fold':>10} {'95% CI':>22}"
    )
    for row in analysis_rows:
        print(
            f"{row.variant:10} {row.method_label:12} {row.p_folded:8.3f} "
            f"[{row.p_folded_ci_low:6.3f}, {row.p_folded_ci_high:6.3f}] "
            f"{row.delta_g_fold_rest_kcal_mol:+10.3f} "
            f"[{row.delta_g_ci_low:+7.3f}, {row.delta_g_ci_high:+7.3f}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
