"""Effective sample size comparison plots for FELP run summaries."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

from felp.core import ConfigurationError

from .common import apply_publication_style, import_matplotlib, prepare_output_dir
from .ess_bootstrap import (
    METADATA_KEYS,
    bootstrap_overlap_metrics,
    percentile_summary,
    prepare_bootstrap_setup,
)


def _ess_compare_bootstrap_pool_worker(
    payload: tuple[str, int, int, int],
) -> dict[str, Any]:
    """Load one run and run block bootstrap in a separate process (ess-compare only)."""

    summary_path, block_size, n_draws, seed = payload
    rng = np.random.default_rng(seed)
    setup = prepare_bootstrap_setup(summary_path, block_size)
    metrics, failed = bootstrap_overlap_metrics(setup, n_draws, rng)
    return {
        "metrics": metrics,
        "failed_draws": failed,
        "run_dir": setup.run_dir,
        "eligible_frame_count": setup.eligible_frame_count,
        "block_count": setup.block_count,
        "block_size": setup.block_size,
    }


def _ess_compare_sequential_only() -> bool:
    return os.environ.get("FELP_ESS_COMPARE_SEQUENTIAL", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _collect_ess_compare_bootstrap_stats(
    input_paths: tuple[str, ...],
    block_size: int,
    n_draws: int,
    base_seed: int,
) -> list[dict[str, Any]]:
    """Prepare setups and run block bootstrap for each input, parallel processes when useful."""

    n = len(input_paths)
    if n > 1 and not _ess_compare_sequential_only():
        payloads = [
            (input_paths[i], block_size, n_draws, int(base_seed) + i * 100_003)
            for i in range(n)
        ]
        max_workers = min(n, max(1, os.cpu_count() or 4))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(_ess_compare_bootstrap_pool_worker, payloads))

    rng = np.random.default_rng(int(base_seed))
    stats: list[dict[str, Any]] = []
    for path in input_paths:
        setup = prepare_bootstrap_setup(path, block_size)
        metrics, failed = bootstrap_overlap_metrics(setup, n_draws, rng)
        stats.append(
            {
                "metrics": metrics,
                "failed_draws": failed,
                "run_dir": setup.run_dir,
                "eligible_frame_count": setup.eligible_frame_count,
                "block_count": setup.block_count,
                "block_size": setup.block_size,
            }
        )
    return stats

_REFERENCE_ORDER = ("GLY", "PRO", "ALA")
_REFERENCE_COLORS = {
    "GLY": "#2d6a4f",  # dark green
    "PRO": "#52b788",  # medium green
    "ALA": "#b7e4c7",  # light green
}
_REFERENCE_EDGE_COLORS = {
    "GLY": "black",
    "PRO": "black",
    "ALA": "black",
}
_DEFAULT_REFERENCE_COLOR = "#666666"
_BAR_EDGE_WIDTH = 0.8
_LOW_ESS_COLORS = ("#40916c", "#95d5b2")  # foresty low ess colors


@dataclass(frozen=True)
class ESSCompareConfig:
    """Configuration for comparing ESS diagnostics across FELP runs."""

    input_paths: tuple[str, ...]
    output_dir: str
    labels: tuple[str, ...] | None = None
    overwrite: bool = False
    bootstrap_block_size: int | None = None
    bootstrap_draws: int | None = None
    bootstrap_seed: int = 0

    def validate(self) -> None:
        """Validate input paths and normalize directories to run summary files."""

        if len(self.input_paths) < 2:
            raise ConfigurationError("ess-compare requires at least two inputs.")
        if not self.output_dir:
            raise ConfigurationError("An output directory is required.")
        if self.labels is not None and len(self.labels) != len(self.input_paths):
            raise ConfigurationError("--labels must match the number of --input values.")

        if (self.bootstrap_block_size is None) ^ (self.bootstrap_draws is None):
            raise ConfigurationError(
                "Provide both --bootstrap-blocks and --bootstrap-draws together, "
                "or omit both."
            )
        if self.bootstrap_block_size is not None:
            if self.bootstrap_block_size < 1:
                raise ConfigurationError("--bootstrap-blocks must be at least 1.")
            if self.bootstrap_draws is not None and self.bootstrap_draws < 1:
                raise ConfigurationError("--bootstrap-draws must be at least 1.")

        summary_paths: list[str] = []
        auto_labels: list[str] = []
        for raw_path in self.input_paths:
            path = os.fspath(raw_path)
            if os.path.isdir(path):
                summary_path = os.path.join(path, "run_summary.json")
                label = os.path.basename(os.path.abspath(path))
            else:
                summary_path = path
                parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
                label = (
                    parent
                    if os.path.basename(path) == "run_summary.json"
                    else os.path.basename(path)
                )
            if not os.path.isfile(summary_path):
                raise ConfigurationError(
                    f"ESS comparison input does not exist: {summary_path}"
                )
            summary_paths.append(summary_path)
            auto_labels.append(label or summary_path)

        object.__setattr__(self, "input_paths", tuple(summary_paths))
        if self.labels is None:
            object.__setattr__(self, "labels", tuple(_unique_labels(auto_labels)))
        else:
            object.__setattr__(self, "labels", tuple(self.labels))


def generate_ess_comparison_plots(config: ESSCompareConfig) -> dict[str, object]:
    """Generate multi-run ESS/overlap comparison plots from run summaries."""

    config.validate()
    prepare_output_dir(config.output_dir, overwrite=config.overwrite)
    plt = import_matplotlib()
    _apply_ess_compare_style(plt)

    rows = [
        _comparison_row(path, label)
        for path, label in zip(config.input_paths, config.labels or (), strict=True)
    ]
    csv_path = os.path.join(config.output_dir, "ess_comparison_summary.csv")
    _write_comparison_csv(csv_path, rows)

    bootstrap_payload: dict[str, object] = {}
    if config.bootstrap_block_size is not None and config.bootstrap_draws is not None:
        bootstrap_stats = _collect_ess_compare_bootstrap_stats(
            config.input_paths,
            int(config.bootstrap_block_size),
            int(config.bootstrap_draws),
            int(config.bootstrap_seed),
        )
        boot_csv = os.path.join(config.output_dir, "ess_comparison_bootstrap.csv")
        _write_bootstrap_csv(
            boot_csv,
            labels=list(config.labels or ()),
            stats=bootstrap_stats,
        )
        bootstrap_payload = {
            "bootstrap_summary_csv": boot_csv,
            "bootstrap_block_size": int(config.bootstrap_block_size),
            "bootstrap_draws": int(config.bootstrap_draws),
            "bootstrap_seed": int(config.bootstrap_seed),
            "bootstrap_failed_draws": [s["failed_draws"] for s in bootstrap_stats],
        }
        n_draw = int(config.bootstrap_draws)
        for i, st in enumerate(bootstrap_stats):
            if st["failed_draws"] >= n_draw:
                raise ConfigurationError(
                    "Block bootstrap failed for every draw for one or more runs. "
                    f"First failing input index {i}, path {config.input_paths[i]!r}. "
                    "Typical causes: reweighting QC filtering removes all frames in "
                    "many resamples, or an outdated FELP build missing bootstrap "
                    "empirical-prior handling for MSM bundle runs."
                )
        outputs = [
            _plot_single_metric_bootstrap(
                plt=plt,
                output_dir=config.output_dir,
                labels=list(config.labels or ()),
                stats=bootstrap_stats,
                key_index=0,
                ylabel="Global ESS fraction",
                title="Global effective sample size fraction (bootstrap)",
                filename="global_ess_fraction.png",
            ),
            _plot_single_metric_bootstrap(
                plt=plt,
                output_dir=config.output_dir,
                labels=list(config.labels or ()),
                stats=bootstrap_stats,
                key_index=1,
                ylabel="Overlap coefficient",
                title="Discrete overlap coefficient (bootstrap)",
                filename="overlap_coefficient.png",
            ),
            _plot_single_metric_bootstrap(
                plt=plt,
                output_dir=config.output_dir,
                labels=list(config.labels or ()),
                stats=bootstrap_stats,
                key_index=2,
                ylabel="Target probability mass",
                title=(
                    "Target probability in states with unreliable local support "
                    "(bootstrap)"
                ),
                filename="unreliable_local_support_mass.png",
            ),
            _plot_single_metric_bootstrap(
                plt=plt,
                output_dir=config.output_dir,
                labels=list(config.labels or ()),
                stats=bootstrap_stats,
                key_index=3,
                ylabel="Max normalized frame weight",
                title="Maximum normalized importance weight (bootstrap)",
                filename="max_normalized_weight.png",
            ),
        ]
        generated_files = [csv_path, boot_csv, *outputs]
    else:
        outputs = [
            _plot_single_metric(
                plt=plt,
                output_dir=config.output_dir,
                rows=rows,
                key="global_effective_sample_fraction",
                ylabel="Global ESS fraction",
                title="Global effective sample size fraction",
                filename="global_ess_fraction.png",
            ),
            _plot_single_metric(
                plt=plt,
                output_dir=config.output_dir,
                rows=rows,
                key="overlap_coefficient",
                ylabel="Overlap coefficient",
                title="Discrete overlap coefficient",
                filename="overlap_coefficient.png",
            ),
            _plot_single_metric(
                plt=plt,
                output_dir=config.output_dir,
                rows=rows,
                key="target_probability_unreliable_local_support",
                ylabel="Target probability mass",
                title=(
                    "Target probability in states with unreliable local support "
                    r"($\mathrm{ESS}_i < \max(\mathrm{ESS}_{\min}, 0.1\,n_i)$)"
                ),
                filename="unreliable_local_support_mass.png",
            ),
            _plot_single_metric(
                plt=plt,
                output_dir=config.output_dir,
                rows=rows,
                key="max_normalized_weight",
                ylabel="Max normalized frame weight",
                title="Maximum normalized importance weight",
                filename="max_normalized_weight.png",
            ),
        ]
        generated_files = [csv_path, *outputs]
    result: dict[str, object] = {
        "output_dir": config.output_dir,
        "input_paths": list(config.input_paths),
        "labels": list(config.labels or ()),
        "comparison_summary_csv": csv_path,
        "generated_files": generated_files,
    }
    result.update(bootstrap_payload)
    return result


def _overlap_diagnostics_for_comparison(summary: dict[str, Any]) -> dict[str, Any]:
    """Return overlap diagnostics from nested or flat per-method run summaries."""

    nested = _mapping(summary.get("overlap_diagnostics"))
    if nested.get("global_effective_sample_fraction") is not None or nested.get(
        "overlap_coefficient"
    ) is not None:
        return nested
    if summary.get("global_effective_sample_fraction") is None and summary.get(
        "overlap_coefficient"
    ) is None:
        return nested
    return {
        "global_effective_sample_size": summary.get("global_effective_sample_size"),
        "global_effective_sample_fraction": summary.get(
            "global_effective_sample_fraction"
        ),
        "overlap_coefficient": summary.get("overlap_coefficient"),
        "max_normalized_weight": summary.get("max_normalized_weight"),
        "top_weight_mass_by_fraction": {
            "top_1_percent": summary.get("top_1_percent_weight_mass"),
            "top_5_percent": summary.get("top_5_percent_weight_mass"),
            "top_10_percent": summary.get("top_10_percent_weight_mass"),
        },
        "target_probability_unreliable_local_support": summary.get(
            "target_probability_unreliable_local_support"
        ),
    }


def _comparison_row(path: str, label: str) -> dict[str, object]:
    with open(path, encoding="utf-8") as handle:
        summary = json.load(handle)
    if not isinstance(summary, dict):
        raise ConfigurationError(f"Run summary must be a JSON object: {path}")

    diagnostics = _overlap_diagnostics_for_comparison(summary)
    top_weight_mass = _mapping(diagnostics.get("top_weight_mass_by_fraction"))
    return {
        "label": label,
        "source_path": path,
        "run_output_dir": summary.get("output_dir", ""),
        "overlap_assessment": str(summary.get("overlap_assessment") or "unknown"),
        "global_effective_sample_size": _float_or_nan(
            diagnostics.get("global_effective_sample_size")
        ),
        "global_effective_sample_fraction": _float_or_nan(
            diagnostics.get("global_effective_sample_fraction")
        ),
        "overlap_coefficient": _float_or_nan(diagnostics.get("overlap_coefficient")),
        "max_normalized_weight": _float_or_nan(diagnostics.get("max_normalized_weight")),
        "top_1_percent_weight_mass": _float_or_nan(
            top_weight_mass.get("top_1_percent")
        ),
        "top_5_percent_weight_mass": _float_or_nan(
            top_weight_mass.get("top_5_percent")
        ),
        "top_10_percent_weight_mass": _float_or_nan(
            top_weight_mass.get("top_10_percent")
        ),
        "target_probability_unreliable_local_support": _float_or_nan(
            diagnostics.get("target_probability_unreliable_local_support")
        ),
    }


def _write_comparison_csv(path: str, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "label",
        "source_path",
        "run_output_dir",
        "overlap_assessment",
        "global_effective_sample_size",
        "global_effective_sample_fraction",
        "overlap_coefficient",
        "max_normalized_weight",
        "top_1_percent_weight_mass",
        "top_5_percent_weight_mass",
        "top_10_percent_weight_mass",
        "target_probability_unreliable_local_support",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_bootstrap_csv(
    path: str,
    *,
    labels: list[str],
    stats: list[dict[str, object]],
) -> None:
    fieldnames = [
        "label",
        "run_dir",
        "eligible_frame_count",
        "block_count",
        "block_size",
        "bootstrap_draws",
        "failed_draws",
    ]
    for key in METADATA_KEYS:
        fieldnames.extend([f"{key}_median", f"{key}_p2_5", f"{key}_p97_5"])

    out_rows: list[dict[str, object]] = []
    for label, item in zip(labels, stats, strict=True):
        metrics: np.ndarray = item["metrics"]
        row: dict[str, object] = {
            "label": label,
            "run_dir": item["run_dir"],
            "eligible_frame_count": item["eligible_frame_count"],
            "block_count": item["block_count"],
            "block_size": item["block_size"],
            "bootstrap_draws": int(metrics.shape[0]),
            "failed_draws": item["failed_draws"],
        }
        for col, key in enumerate(METADATA_KEYS):
            med, lo, hi = percentile_summary(metrics[:, col])
            row[f"{key}_median"] = med
            row[f"{key}_p2_5"] = lo
            row[f"{key}_p97_5"] = hi
        out_rows.append(row)

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)


def _asymmetric_yerr(med: float, lo: float, hi: float) -> tuple[float, float]:
    if not all(math.isfinite(v) for v in (med, lo, hi)):
        return 0.0, 0.0
    return max(0.0, med - lo), max(0.0, hi - med)


def _reference_from_label(label: str) -> str:
    """Extract the reference residue from labels like ``GLY→ALA`` or run dir names."""

    text = str(label).strip()
    for separator in ("→", "->", "—", "-->"):
        if separator in text:
            return text.split(separator, 1)[0].strip().upper()
    match = re.search(r"felp_([A-Za-z]+)_to_", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return "UNKNOWN"


def _target_from_label(label: str) -> str:
    """Extract the mutation target from a comparison label."""

    text = str(label).strip()
    for separator in ("→", "->", "—", "-->"):
        if separator in text:
            return text.split(separator, 1)[1].strip().upper()
    match = re.search(r"felp_[A-Za-z]+_to_([A-Za-z]+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return text


def _reference_color(reference: str) -> tuple[str, str]:
    """Return face and edge colors for a reference residue."""

    ref = reference.upper()
    return (
        _REFERENCE_COLORS.get(ref, _DEFAULT_REFERENCE_COLOR),
        _REFERENCE_EDGE_COLORS.get(ref, "#333333"),
    )


def _group_indices_by_reference(labels: list[str]) -> dict[str, list[int]]:
    """Group comparison indices by reference residue, targets sorted alphabetically."""

    buckets: dict[str, list[tuple[str, int]]] = {}
    for index, label in enumerate(labels):
        reference = _reference_from_label(label)
        target = _target_from_label(label)
        buckets.setdefault(reference, []).append((target, index))
    grouped: dict[str, list[int]] = {}
    for reference, items in buckets.items():
        grouped[reference] = [index for _, index in sorted(items, key=lambda pair: pair[0])]
    return grouped


def _ordered_references(grouped: dict[str, list[int]]) -> list[str]:
    """Stable panel order: GLY, PRO, ALA, then any others alphabetically."""

    extras = sorted(reference for reference in grouped if reference not in _REFERENCE_ORDER)
    return [reference for reference in (*_REFERENCE_ORDER, *extras) if reference in grouped]


def _tick_fontsize(n_bars: int) -> float:
    """Scale x tick labels so long mutation lists remain readable."""

    if n_bars <= 12:
        return 12.0
    if n_bars <= 20:
        return 11.0
    return max(8.0, 11.0 - 0.1 * (n_bars - 20))


def _apply_ess_compare_style(plt) -> None:
    """Matplotlib rcParams tuned for multi-panel ESS comparison figures."""

    apply_publication_style()
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
        }
    )


def _save_figure(fig, path: str) -> str:
    """Write PNG and vector PDF for publication."""

    fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
    pdf_path = os.path.splitext(path)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08)
    return path


def _build_grouped_bar_layout(
    labels: list[str],
) -> tuple[list[str], list[str], dict[str, dict[str, int]]]:
    """Build a target-grouped layout mapping.

    Returns (sorted_targets, ordered_references, mapping) where
    ``mapping[target][reference] = original_index``.
    """

    grouped = _group_indices_by_reference(labels)
    references = _ordered_references(grouped)
    target_map: dict[str, dict[str, int]] = {}
    for reference in references:
        for index in grouped[reference]:
            target = _target_from_label(labels[index])
            target_map.setdefault(target, {})[reference] = index
    sorted_targets = sorted(target_map)
    return sorted_targets, references, target_map


def _plot_single_metric_bootstrap(
    *,
    plt,
    output_dir: str,
    labels: list[str],
    stats: list[dict[str, object]],
    key_index: int,
    ylabel: str,
    title: str,
    filename: str,
) -> str:
    targets, references, target_map = _build_grouped_bar_layout(labels)
    n_targets = len(targets)
    n_refs = len(references)
    bar_width = 0.8 / max(n_refs, 1)
    x = np.arange(n_targets)
    fig_width = max(8.0, 0.6 * n_targets * n_refs + 2.0)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 5.5))
    y_highs: list[float] = []

    for ref_idx, reference in enumerate(references):
        face, edge = _reference_color(reference)
        offset = (ref_idx - (n_refs - 1) / 2.0) * bar_width
        medians: list[float] = []
        yerr_lo: list[float] = []
        yerr_hi: list[float] = []
        for target in targets:
            index = target_map.get(target, {}).get(reference)
            if index is not None:
                metrics: np.ndarray = stats[index]["metrics"]
                med, lo, hi = percentile_summary(metrics[:, key_index])
            else:
                med, lo, hi = float("nan"), float("nan"), float("nan")
            medians.append(med)
            el, eh = _asymmetric_yerr(med, lo, hi)
            yerr_lo.append(el)
            yerr_hi.append(eh)
            if math.isfinite(med):
                y_highs.append(med + eh)

        ax.bar(
            x + offset,
            medians,
            bar_width * 0.92,
            label=reference,
            color=face,
            edgecolor=edge,
            linewidth=_BAR_EDGE_WIDTH,
            yerr=[yerr_lo, yerr_hi],
            capsize=2,
            ecolor=edge,
            error_kw={"linewidth": 0.8},
        )

    ax.set_title(title, fontsize=16, pad=12)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(targets, fontsize=_tick_fontsize(n_targets))
    ax.tick_params(axis="x", length=0, pad=4)
    ax.legend(frameon=False, fontsize=12)
    _set_metric_ylim(ax, y_highs)
    fig.tight_layout()
    path = os.path.join(output_dir, filename)
    _save_figure(fig, path)
    plt.close(fig)
    return path


def _plot_single_metric(
    *,
    plt,
    output_dir: str,
    rows: list[dict[str, object]],
    key: str,
    ylabel: str,
    title: str,
    filename: str,
) -> str:
    labels = [str(row["label"]) for row in rows]
    values = [_float_or_nan(row.get(key)) for row in rows]
    targets, references, target_map = _build_grouped_bar_layout(labels)
    n_targets = len(targets)
    n_refs = len(references)
    bar_width = 0.8 / max(n_refs, 1)
    x = np.arange(n_targets)
    fig_width = max(8.0, 0.6 * n_targets * n_refs + 2.0)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 5.5))

    for ref_idx, reference in enumerate(references):
        face, edge = _reference_color(reference)
        offset = (ref_idx - (n_refs - 1) / 2.0) * bar_width
        bar_values = [
            values[target_map[t][reference]]
            if reference in target_map.get(t, {})
            else float("nan")
            for t in targets
        ]
        ax.bar(
            x + offset,
            bar_values,
            bar_width * 0.92,
            label=reference,
            color=face,
            edgecolor=edge,
            linewidth=_BAR_EDGE_WIDTH,
        )

    ax.set_title(title, fontsize=16, pad=12)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(targets, fontsize=_tick_fontsize(n_targets))
    ax.tick_params(axis="x", length=0, pad=4)
    ax.legend(frameon=False, fontsize=12)
    _set_metric_ylim(ax, values)
    fig.tight_layout()
    path = os.path.join(output_dir, filename)
    _save_figure(fig, path)
    plt.close(fig)
    return path


def _set_metric_ylim(
    ax,
    values: list[float],
) -> None:
    finite_values = [value for value in values if math.isfinite(value)]
    ymax = max([*finite_values, 1.0e-12])
    ax.set_ylim(bottom=0.0, top=ymax * 1.15)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _float_or_nan(value: Any) -> float:
    if value in (None, ""):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _unique_labels(labels: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    unique: list[str] = []
    for label in labels:
        count = counts.get(label, 0) + 1
        counts[label] = count
        unique.append(label if count == 1 else f"{label}_{count}")
    return unique
