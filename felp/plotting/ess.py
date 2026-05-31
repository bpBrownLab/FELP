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


def _ess_compare_bootstrap_max_workers(n_comparisons: int) -> int:
    """Worker count for one bootstrap job per reference→target comparison."""

    env = os.environ.get("FELP_ESS_COMPARE_MAX_WORKERS", "").strip()
    if env:
        return max(1, min(n_comparisons, int(env)))
    cpu = os.cpu_count() or 4
    return max(1, min(n_comparisons, cpu))


def _ess_compare_bootstrap_chunksize(n_comparisons: int, max_workers: int) -> int:
    """Balance ProcessPoolExecutor scheduling overhead against load balance."""

    if max_workers <= 0:
        return 1
    return max(1, n_comparisons // (max_workers * 4))


def _collect_ess_compare_bootstrap_stats(
    input_paths: tuple[str, ...],
    block_size: int,
    n_draws: int,
    base_seed: int,
) -> list[dict[str, Any]]:
    """Prepare setups and run block bootstrap for each input, parallel processes when useful."""

    import sys

    n = len(input_paths)
    if n > 1 and not _ess_compare_sequential_only():
        payloads = [
            (input_paths[i], block_size, n_draws, int(base_seed) + i * 100_003)
            for i in range(n)
        ]
        max_workers = _ess_compare_bootstrap_max_workers(n)
        chunksize = _ess_compare_bootstrap_chunksize(n, max_workers)
        print(
            "ess-compare bootstrap: "
            f"{n} reference→target runs, {max_workers} workers, "
            f"{n_draws} draws/run, block_size={block_size}, chunksize={chunksize}",
            file=sys.stderr,
        )
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            return list(
                executor.map(
                    _ess_compare_bootstrap_pool_worker,
                    payloads,
                    chunksize=chunksize,
                )
            )

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

_REFERENCE_ORDER = ("GLY", "ALA", "PRO")
_REFERENCE_COLORS = {
    "GLY": "#2A533E",  # deep pine green
    "ALA": "#537F80",  # slate eucalyptus blue-green
    "PRO": "#A36049",  # warm terracotta/ochre
}
_REFERENCE_EDGE_COLORS = {
    "GLY": "black",
    "PRO": "black",
    "ALA": "black",
}
_DEFAULT_REFERENCE_COLOR = "#52796f"
_BAR_EDGE_WIDTH = 0.8
_PANEL_LABEL_SIZE = 14.0
_PANEL_LABEL_WEIGHT = "bold"
_REFERENCE_HEADER_SIZE = 11.5
_TWO_PANEL_FIGURE = "overlap_diagnostics_two_panel.png"
_TWO_PANEL_YLABEL_SIZE = 9.25
_STANDARD_AMINO_ACIDS = frozenset(
    {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    }
)


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
    panel_layout: str = "by_reference"

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
        if self.panel_layout not in {"combined", "by_reference"}:
            raise ConfigurationError(
                "panel_layout must be 'combined' or 'by_reference'."
            )

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

    labels = list(config.labels or ())
    input_paths = list(config.input_paths)
    keep = [
        index
        for index, label in enumerate(labels)
        if _target_from_label(label) in _STANDARD_AMINO_ACIDS
    ]
    labels = [labels[index] for index in keep]
    input_paths = [input_paths[index] for index in keep]
    if len(input_paths) < 2:
        raise ConfigurationError(
            "Fewer than two comparisons remain after restricting targets to the "
            "standard 20 amino acids."
        )

    rows = [
        _comparison_row(path, label)
        for path, label in zip(input_paths, labels, strict=True)
    ]
    csv_path = os.path.join(config.output_dir, "ess_comparison_summary.csv")
    _write_comparison_csv(csv_path, rows)

    bootstrap_payload: dict[str, object] = {}
    if config.bootstrap_block_size is not None and config.bootstrap_draws is not None:
        bootstrap_stats = _collect_ess_compare_bootstrap_stats(
            tuple(input_paths),
            int(config.bootstrap_block_size),
            int(config.bootstrap_draws),
            int(config.bootstrap_seed),
        )
        boot_csv = os.path.join(config.output_dir, "ess_comparison_bootstrap.csv")
        _write_bootstrap_csv(
            boot_csv,
            labels=labels,
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
                    f"First failing input index {i}, path {input_paths[i]!r}. "
                    "Typical causes: reweighting QC filtering removes all frames in "
                    "many resamples, or an outdated FELP build missing bootstrap "
                    "empirical-prior handling for MSM bundle runs."
                )
        plot_kwargs = {
            "plt": plt,
            "output_dir": config.output_dir,
            "labels": labels,
            "stats": bootstrap_stats,
            "panel_layout": config.panel_layout,
        }
        outputs = [
            _plot_single_metric_bootstrap(
                **plot_kwargs,
                key_index=0,
                ylabel="Global ESS fraction",
                title="Global effective sample size fraction (bootstrap)",
                filename="global_ess_fraction.png",
            ),
            _plot_single_metric_bootstrap(
                **plot_kwargs,
                key_index=1,
                ylabel="Overlap coefficient",
                title="Discrete overlap coefficient (bootstrap)",
                filename="overlap_coefficient.png",
            ),
            _plot_single_metric_bootstrap(
                **plot_kwargs,
                key_index=2,
                ylabel="Target probability mass",
                title=(
                    "Target probability in states with unreliable local support "
                    "(bootstrap)"
                ),
                filename="unreliable_local_support_mass.png",
            ),
            _plot_single_metric_bootstrap(
                **plot_kwargs,
                key_index=3,
                ylabel="Max normalized frame weight",
                title="Maximum normalized importance weight (bootstrap)",
                filename="max_normalized_weight.png",
            ),
            _plot_single_metric_bootstrap(
                **plot_kwargs,
                key_index=4,
                ylabel=r"Target probability mass ($\mathrm{ESS}_i < 10$)",
                title=r"Target probability in states with local $\mathrm{ESS}_i < 10$ (bootstrap)",
                filename="unreliable_local_support_mass_ess_lt_10.png",
            ),
            _plot_single_metric_bootstrap(
                **plot_kwargs,
                key_index=5,
                ylabel=r"Target probability mass ($\mathrm{ESS}_i < 50$)",
                title=r"Target probability in states with local $\mathrm{ESS}_i < 50$ (bootstrap)",
                filename="unreliable_local_support_mass_ess_lt_50.png",
            ),
        ]
        two_panel = _plot_overlap_diagnostics_two_panel(
            plt=plt,
            output_dir=config.output_dir,
            labels=labels,
            stats=bootstrap_stats,
            use_bootstrap=True,
        )
        generated_files = [csv_path, boot_csv, two_panel, *outputs]
    else:
        plot_kwargs = {
            "plt": plt,
            "output_dir": config.output_dir,
            "rows": rows,
            "panel_layout": config.panel_layout,
        }
        outputs = [
            _plot_single_metric(
                **plot_kwargs,
                key="global_effective_sample_fraction",
                ylabel="Global ESS fraction",
                title="Global effective sample size fraction",
                filename="global_ess_fraction.png",
            ),
            _plot_single_metric(
                **plot_kwargs,
                key="overlap_coefficient",
                ylabel="Overlap coefficient",
                title="Discrete overlap coefficient",
                filename="overlap_coefficient.png",
            ),
            _plot_single_metric(
                **plot_kwargs,
                key="target_probability_unreliable_local_support",
                ylabel="Target probability mass",
                title=(
                    "Target probability in states with unreliable local support "
                    r"($\mathrm{ESS}_i < \max(\mathrm{ESS}_{\min}, 0.1\,n_i)$)"
                ),
                filename="unreliable_local_support_mass.png",
            ),
            _plot_single_metric(
                **plot_kwargs,
                key="max_normalized_weight",
                ylabel="Max normalized frame weight",
                title="Maximum normalized importance weight",
                filename="max_normalized_weight.png",
            ),
            _plot_single_metric(
                **plot_kwargs,
                key="target_probability_low_ess_mass_ess_lt_10",
                ylabel=r"Target probability mass ($\mathrm{ESS}_i < 10$)",
                title=r"Target probability in states with local $\mathrm{ESS}_i < 10$",
                filename="unreliable_local_support_mass_ess_lt_10.png",
            ),
            _plot_single_metric(
                **plot_kwargs,
                key="target_probability_low_ess_mass_ess_lt_50",
                ylabel=r"Target probability mass ($\mathrm{ESS}_i < 50$)",
                title=r"Target probability in states with local $\mathrm{ESS}_i < 50$",
                filename="unreliable_local_support_mass_ess_lt_50.png",
            ),
        ]
        two_panel = _plot_overlap_diagnostics_two_panel(
            plt=plt,
            output_dir=config.output_dir,
            rows=rows,
            use_bootstrap=False,
        )
        generated_files = [csv_path, two_panel, *outputs]
    result: dict[str, object] = {
        "output_dir": config.output_dir,
        "input_paths": input_paths,
        "labels": labels,
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
        "target_probability_low_ess_mass_ess_lt_10": summary.get(
            "target_probability_low_ess_mass_ess_lt_10"
        ),
        "target_probability_low_ess_mass_ess_lt_50": summary.get(
            "target_probability_low_ess_mass_ess_lt_50"
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
        "target_probability_low_ess_mass_ess_lt_10": _float_or_nan(
            diagnostics.get("target_probability_low_ess_mass_ess_lt_10")
        ),
        "target_probability_low_ess_mass_ess_lt_50": _float_or_nan(
            diagnostics.get("target_probability_low_ess_mass_ess_lt_50")
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
        "target_probability_low_ess_mass_ess_lt_10",
        "target_probability_low_ess_mass_ess_lt_50",
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
        "black",
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
    """Stable panel order: GLY, ALA, PRO, then any others alphabetically."""

    extras = sorted(reference for reference in grouped if reference not in _REFERENCE_ORDER)
    return [reference for reference in (*_REFERENCE_ORDER, *extras) if reference in grouped]


def _tick_fontsize(n_bars: int) -> float:
    """Scale x tick labels so long mutation lists remain readable."""

    if n_bars <= 5:
        return 9.5
    if n_bars <= 12:
        return 8.0
    if n_bars <= 20:
        return 7.0
    return max(5.5, 7.0 - 0.1 * (n_bars - 20))


def _apply_ess_compare_style(plt) -> None:
    """Matplotlib rcParams tuned for multi-panel ESS comparison figures."""

    apply_publication_style()
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 9.5,
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
    panel_layout: str = "combined",
) -> str:
    if panel_layout == "by_reference":
        return _plot_single_metric_bootstrap_by_reference(
            plt=plt,
            output_dir=output_dir,
            labels=labels,
            stats=stats,
            key_index=key_index,
            ylabel=ylabel,
            title=title,
            filename=filename,
        )

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
    ax.set_xticklabels(
        targets,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=_tick_fontsize(n_targets),
    )
    ax.tick_params(axis="x", length=0, pad=4)
    ax.legend(frameon=False, fontsize=12)
    _set_metric_ylim(ax, y_highs, filename)
    fig.tight_layout()
    path = os.path.join(output_dir, filename)
    _save_figure(fig, path)
    plt.close(fig)
    return path


def _plot_single_metric_bootstrap_by_reference(
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
    grouped = _group_indices_by_reference(labels)
    references = _ordered_references(grouped)
    n_refs = len(references)
    fig_width = max(8.0, 4.0 * n_refs)
    fig, axes = plt.subplots(1, n_refs, figsize=(fig_width, 5.5), sharey=True)
    if n_refs == 1:
        axes = [axes]
    y_highs: list[float] = []

    for ax, reference in zip(axes, references, strict=True):
        indices = grouped[reference]
        targets = [_target_from_label(labels[index]) for index in indices]
        face, edge = _reference_color(reference)
        medians: list[float] = []
        yerr_lo: list[float] = []
        yerr_hi: list[float] = []
        for index in indices:
            metrics: np.ndarray = stats[index]["metrics"]
            med, lo, hi = percentile_summary(metrics[:, key_index])
            medians.append(med)
            el, eh = _asymmetric_yerr(med, lo, hi)
            yerr_lo.append(el)
            yerr_hi.append(eh)
            if math.isfinite(med):
                y_highs.append(med + eh)

        x = np.arange(len(targets))
        ax.bar(
            x,
            medians,
            0.72,
            color=face,
            edgecolor=edge,
            linewidth=_BAR_EDGE_WIDTH,
            yerr=[yerr_lo, yerr_hi],
            capsize=2,
            ecolor=edge,
            error_kw={"linewidth": 0.8},
        )
        ax.set_title(f"{reference} reference", fontsize=14, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            targets,
            rotation=45,
            ha="right",
            rotation_mode="anchor",
            fontsize=_tick_fontsize(len(targets)),
        )
        ax.tick_params(axis="x", length=0, pad=4)

    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=16, y=1.02)
    for ax in axes:
        _set_metric_ylim(ax, y_highs, filename)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
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
    panel_layout: str = "combined",
) -> str:
    labels = [str(row["label"]) for row in rows]
    values = [_float_or_nan(row.get(key)) for row in rows]
    if panel_layout == "by_reference":
        return _plot_single_metric_by_reference(
            plt=plt,
            output_dir=output_dir,
            labels=labels,
            values=values,
            ylabel=ylabel,
            title=title,
            filename=filename,
        )

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
    ax.set_xticklabels(
        targets,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=_tick_fontsize(n_targets),
    )
    ax.tick_params(axis="x", length=0, pad=4)
    ax.legend(frameon=False, fontsize=12)
    _set_metric_ylim(ax, values, filename)
    fig.tight_layout()
    path = os.path.join(output_dir, filename)
    _save_figure(fig, path)
    plt.close(fig)
    return path


def _plot_single_metric_by_reference(
    *,
    plt,
    output_dir: str,
    labels: list[str],
    values: list[float],
    ylabel: str,
    title: str,
    filename: str,
) -> str:
    grouped = _group_indices_by_reference(labels)
    references = _ordered_references(grouped)
    n_refs = len(references)
    fig_width = max(8.0, 4.0 * n_refs)
    fig, axes = plt.subplots(1, n_refs, figsize=(fig_width, 5.5), sharey=True)
    if n_refs == 1:
        axes = [axes]

    for ax, reference in zip(axes, references, strict=True):
        indices = grouped[reference]
        targets = [_target_from_label(labels[index]) for index in indices]
        bar_values = [values[index] for index in indices]
        face, edge = _reference_color(reference)
        x = np.arange(len(targets))
        ax.bar(
            x,
            bar_values,
            0.72,
            color=face,
            edgecolor=edge,
            linewidth=_BAR_EDGE_WIDTH,
        )
        ax.set_title(f"{reference} reference", fontsize=14, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            targets,
            rotation=45,
            ha="right",
            rotation_mode="anchor",
            fontsize=_tick_fontsize(len(targets)),
        )
        ax.tick_params(axis="x", length=0, pad=4)

    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=16, y=1.02)
    for ax in axes:
        _set_metric_ylim(ax, values, filename)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, filename)
    _save_figure(fig, path)
    plt.close(fig)
    return path


def _plot_overlap_diagnostics_two_panel(
    *,
    plt,
    output_dir: str,
    labels: list[str] | None = None,
    rows: list[dict[str, object]] | None = None,
    stats: list[dict[str, object]] | None = None,
    use_bootstrap: bool,
) -> str:
    """Publication two-panel figure: (A) ESS_i<10 mass, (B) max normalized weight."""

    if use_bootstrap:
        if labels is None or stats is None:
            raise ValueError("Bootstrap two-panel figure requires labels and stats.")
        panel_specs = (
            (
                "A",
                4,
                "target_probability_low_ess_mass_ess_lt_10",
                r"Target probability mass ($\mathrm{ESS}_i < 10$)",
                "linear",
            ),
            (
                "B",
                3,
                "max_normalized_weight",
                "Max normalized frame weight",
                "log_max_weight",
            ),
        )

        def stats_for_index(index: int, key_index: int) -> tuple[float, float, float]:
            metrics: np.ndarray = stats[index]["metrics"]
            return percentile_summary(metrics[:, key_index])

    else:
        if rows is None:
            raise ValueError("Non-bootstrap two-panel figure requires rows.")
        labels = [str(row["label"]) for row in rows]
        panel_specs = (
            (
                "A",
                "target_probability_low_ess_mass_ess_lt_10",
                r"Target probability mass ($\mathrm{ESS}_i < 10$)",
                "linear",
            ),
            (
                "B",
                "max_normalized_weight",
                "Max normalized frame weight",
                "log_max_weight",
            ),
        )

        def value_for_index(index: int, key: str) -> float:
            return _float_or_nan(rows[index].get(key))

    grouped = _group_indices_by_reference(labels)
    references = _ordered_references(grouped)
    n_refs = len(references)
    n_targets = max(len(grouped[reference]) for reference in references)
    tick_size = _tick_fontsize(n_targets)

    # Coordinated figure size: generous width for high-density bar spacing
    fig_width = max(11.0, 3.8 * n_refs)
    fig_height = 6.4
    fig, axes = plt.subplots(2, n_refs, figsize=(fig_width, fig_height), sharex="col")
    if n_refs == 1:
        axes = np.asarray([[axes[0]], [axes[1]]])

    for row_idx, spec in enumerate(panel_specs):
        panel_label = spec[0]
        ylabel = spec[-2]
        scale_kind = spec[-1]
        row_values: list[float] = []
        reference_bars: list[
            tuple[
                list[str],
                list[float],
                str,
                list[float] | None,
                list[float] | None,
            ]
        ] = []

        for reference in references:
            indices = grouped[reference]
            targets = [_target_from_label(labels[index]) for index in indices]
            yerr_lo: list[float] | None = None
            yerr_hi: list[float] | None = None
            if use_bootstrap:
                key_index = int(spec[1])
                bar_values = []
                yerr_lo = []
                yerr_hi = []
                for index in indices:
                    med, lo, hi = stats_for_index(index, key_index)
                    bar_values.append(med)
                    el, eh = _asymmetric_yerr(med, lo, hi)
                    yerr_lo.append(el)
                    yerr_hi.append(eh)
                    if math.isfinite(med):
                        row_values.append(med + eh)
                        if scale_kind == "log_max_weight" and el > 0:
                            row_values.append(max(med - el, 1.0e-12))
            else:
                key = str(spec[1])
                bar_values = [value_for_index(index, key) for index in indices]
                row_values.extend(bar_values)
            reference_bars.append((targets, bar_values, reference, yerr_lo, yerr_hi))

        for col_idx, (targets, bar_values, reference, yerr_lo, yerr_hi) in enumerate(
            reference_bars
        ):
            ax = axes[row_idx, col_idx]
            ax.set_axisbelow(True)  # Grid lines go strictly behind the bars
            face, edge = _reference_color(reference)
            x = np.arange(len(targets))
            bar_kwargs: dict[str, object] = {
                "width": 0.78,
                "color": face,
                "edgecolor": edge,
                "linewidth": 0.5,
            }
            if yerr_lo is not None and yerr_hi is not None:
                bar_kwargs.update(
                    {
                        "yerr": [yerr_lo, yerr_hi],
                        "capsize": 2,
                        "ecolor": "black",
                        "error_kw": {"linewidth": 0.8},
                    }
                )
            ax.bar(x, bar_values, **bar_kwargs)

            # Standardized title positioning via Matplotlib title engine
            if row_idx == 0:
                ax.set_title(reference, loc="center", fontweight="semibold", pad=10)

            ax.set_xticks(x)
            ax.set_xticklabels(
                targets,
                rotation=45,
                ha="right",
                rotation_mode="anchor",
                fontsize=tick_size,
            )
            ax.tick_params(axis="x", length=0, pad=3, labelbottom=True)

            if col_idx == 0:
                ax.set_ylabel(ylabel, labelpad=8, fontsize=_TWO_PANEL_YLABEL_SIZE)
                # Left-aligned panel title via Matplotlib title engine to prevent overlaps
                ax.set_title(f"({panel_label})", loc="left", fontweight="bold", pad=10)

        for col_idx in range(n_refs):
            _apply_metric_ylim(axes[row_idx, col_idx], row_values, scale_kind)

    # Generous spacing to allow full rotation, label clarity, and margins
    fig.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.18, hspace=0.42, wspace=0.14)
    path = os.path.join(output_dir, _TWO_PANEL_FIGURE)
    _save_figure(fig, path)
    plt.close(fig)
    return path


def _apply_metric_ylim(
    ax,
    values: list[float],
    scale_kind: str,
) -> None:
    """Set y limits and grid for a metric row in the two-panel figure."""

    finite_values = [value for value in values if math.isfinite(value) and value > 0]
    if not finite_values:
        finite_values = [1.0e-12]

    if scale_kind == "log_max_weight":
        ax.set_yscale("log")
        ymin = min(finite_values)
        bottom = max(1e-6, 10 ** math.floor(math.log10(ymin))) if ymin < 1.0 else 1e-6
        ax.set_ylim(bottom=bottom, top=1.1)
        ax.grid(True, which="both", axis="y", linestyle=":", alpha=0.35, color="#888888")
    else:
        # Standardized 0.0 to 1.05 bound for probability mass plots (no clipping)
        ax.set_ylim(bottom=0.0, top=1.05)
        ax.grid(True, which="major", axis="y", linestyle=":", alpha=0.35, color="#888888")


def _set_metric_ylim(
    ax,
    values: list[float],
    filename: str = "",
) -> None:
    finite_values = [value for value in values if math.isfinite(value) and value > 0]
    if not finite_values:
        finite_values = [1.0e-12]
    
    if filename == "max_normalized_weight.png":
        ax.set_yscale("log")
        ymin = min(finite_values)
        bottom = max(1e-6, 10 ** math.floor(math.log10(ymin))) if ymin < 1.0 else 1e-6
        ax.set_ylim(bottom=bottom, top=1.1)
        ax.grid(True, which="both", axis="y", linestyle=":", alpha=0.5, color="gray")
    else:
        ymax = max(finite_values)
        ax.set_ylim(bottom=0.0, top=min(1.05, max(ymax * 1.15, 0.05)))
        ax.grid(True, which="major", axis="y", linestyle=":", alpha=0.5, color="gray")


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
