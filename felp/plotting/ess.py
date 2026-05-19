"""Effective sample size comparison plots for FELP run summaries."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
import os
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

_BAR_FACE_COLOR = "#d4d4d4"
_BAR_EDGE_COLOR = "black"
_BAR_EDGE_WIDTH = 0.8


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
    apply_publication_style()

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
            _plot_low_ess_mass_bootstrap(
                plt=plt,
                output_dir=config.output_dir,
                labels=list(config.labels or ()),
                stats=bootstrap_stats,
            ),
            _plot_single_metric_bootstrap(
                plt=plt,
                output_dir=config.output_dir,
                labels=list(config.labels or ()),
                stats=bootstrap_stats,
                key_index=4,
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
            _plot_low_ess_mass(
                plt=plt,
                output_dir=config.output_dir,
                rows=rows,
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


def _comparison_row(path: str, label: str) -> dict[str, object]:
    with open(path, encoding="utf-8") as handle:
        summary = json.load(handle)
    if not isinstance(summary, dict):
        raise ConfigurationError(f"Run summary must be a JSON object: {path}")

    diagnostics = _mapping(summary.get("overlap_diagnostics"))
    low_ess = _mapping(diagnostics.get("target_probability_low_ess_mass_by_threshold"))
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
        "target_probability_low_ess_mass_ess_lt_10": _float_or_nan(
            low_ess.get("ess_lt_10")
        ),
        "target_probability_low_ess_mass_ess_lt_50": _float_or_nan(
            low_ess.get("ess_lt_50")
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
    x = np.arange(len(stats))
    medians: list[float] = []
    yerr_lo: list[float] = []
    yerr_hi: list[float] = []
    for item in stats:
        metrics: np.ndarray = item["metrics"]
        med, lo, hi = percentile_summary(metrics[:, key_index])
        medians.append(med)
        el, eh = _asymmetric_yerr(med, lo, hi)
        yerr_lo.append(el)
        yerr_hi.append(eh)

    fig, ax = plt.subplots(1, 1)
    ax.bar(
        x,
        medians,
        color=_BAR_FACE_COLOR,
        edgecolor=_BAR_EDGE_COLOR,
        linewidth=_BAR_EDGE_WIDTH,
        yerr=[yerr_lo, yerr_hi],
        capsize=3,
        ecolor="black",
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    highs = [medians[i] + yerr_hi[i] for i in range(len(medians))]
    _set_metric_ylim(ax, [*medians, *highs])
    fig.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_low_ess_mass_bootstrap(
    *,
    plt,
    output_dir: str,
    labels: list[str],
    stats: list[dict[str, object]],
) -> str:
    n = len(stats)
    x = np.arange(n)
    width = 0.36
    med10: list[float] = []
    lo10: list[float] = []
    hi10: list[float] = []
    med50: list[float] = []
    lo50: list[float] = []
    hi50: list[float] = []
    for item in stats:
        metrics: np.ndarray = item["metrics"]
        a10, b10, c10 = percentile_summary(metrics[:, 2])
        a50, b50, c50 = percentile_summary(metrics[:, 3])
        med10.append(a10)
        lo10.append(b10)
        hi10.append(c10)
        med50.append(a50)
        lo50.append(b50)
        hi50.append(c50)

    y10_lo = [_asymmetric_yerr(med10[i], lo10[i], hi10[i])[0] for i in range(n)]
    y10_hi = [_asymmetric_yerr(med10[i], lo10[i], hi10[i])[1] for i in range(n)]
    y50_lo = [_asymmetric_yerr(med50[i], lo50[i], hi50[i])[0] for i in range(n)]
    y50_hi = [_asymmetric_yerr(med50[i], lo50[i], hi50[i])[1] for i in range(n)]

    fig, ax = plt.subplots(1, 1)
    ax.bar(
        x - width / 2.0,
        med10,
        width,
        label="ESS < 10",
        color=_BAR_FACE_COLOR,
        edgecolor=_BAR_EDGE_COLOR,
        linewidth=_BAR_EDGE_WIDTH,
        yerr=[y10_lo, y10_hi],
        capsize=3,
        ecolor="black",
    )
    ax.bar(
        x + width / 2.0,
        med50,
        width,
        label="ESS < 50",
        color=_BAR_FACE_COLOR,
        edgecolor=_BAR_EDGE_COLOR,
        linewidth=_BAR_EDGE_WIDTH,
        yerr=[y50_lo, y50_hi],
        capsize=3,
        ecolor="black",
    )
    ax.set_title("Target probability mass in low-ESS states (bootstrap)")
    ax.set_ylabel("Target probability mass")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend()
    finite_hi = [v for v in hi10 + hi50 if math.isfinite(v)]
    ymax = max(finite_hi) if finite_hi else 1.0e-12
    ax.set_ylim(0.0, ymax * 1.15)
    fig.tight_layout()
    path = os.path.join(output_dir, "low_ess_mass.png")
    fig.savefig(path)
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
    values = [_float_or_nan(row.get(key)) for row in rows]
    labels = [str(row["label"]) for row in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(1, 1)
    ax.bar(
        x,
        values,
        color=_BAR_FACE_COLOR,
        edgecolor=_BAR_EDGE_COLOR,
        linewidth=_BAR_EDGE_WIDTH,
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    _set_metric_ylim(ax, values)
    fig.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_low_ess_mass(
    *,
    plt,
    output_dir: str,
    rows: list[dict[str, object]],
) -> str:
    labels = [str(row["label"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.38
    ess_lt_10 = [
        _float_or_nan(row.get("target_probability_low_ess_mass_ess_lt_10"))
        for row in rows
    ]
    ess_lt_50 = [
        _float_or_nan(row.get("target_probability_low_ess_mass_ess_lt_50"))
        for row in rows
    ]
    fig, ax = plt.subplots(1, 1)
    ax.bar(
        x - width / 2.0,
        ess_lt_10,
        width,
        label="ESS < 10",
        color=_BAR_FACE_COLOR,
        edgecolor=_BAR_EDGE_COLOR,
        linewidth=_BAR_EDGE_WIDTH,
    )
    ax.bar(
        x + width / 2.0,
        ess_lt_50,
        width,
        label="ESS < 50",
        color=_BAR_FACE_COLOR,
        edgecolor=_BAR_EDGE_COLOR,
        linewidth=_BAR_EDGE_WIDTH,
    )
    ax.set_title("Target probability mass in low-ESS states")
    ax.set_ylabel("Target probability mass")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend()
    _set_metric_ylim(ax, [*ess_lt_10, *ess_lt_50])
    fig.tight_layout()
    path = os.path.join(output_dir, "low_ess_mass.png")
    fig.savefig(path)
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
