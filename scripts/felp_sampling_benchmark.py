"""Prepare and analyze FELP microstate sampling-density benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shlex
import sys
from typing import Any, Iterable


DEFAULT_FRACTIONS = (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _python_felp_parts() -> tuple[str, str]:
    return sys.executable, str(_repo_root() / "felp.py")


def _format_fraction_tag(fraction: float) -> str:
    if fraction == 1.0:
        return "baseline"
    return f"frac_{fraction:g}".replace(".", "p")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_microstate_summary(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed = dict(row)
            metadata_raw = parsed.get("metadata") or "{}"
            parsed["metadata"] = json.loads(metadata_raw)
            for key in (
                "reference_probability",
                "target_probability",
                "free_energy_kj_mol",
                "effective_sample_size",
                "mean_delta_u_kj_mol",
                "std_delta_u_kj_mol",
                "mean_boltzmann_factor",
                "std_boltzmann_factor",
            ):
                value = parsed.get(key)
                parsed[key] = float(value) if value not in (None, "") else math.nan
            rows.append(parsed)
    return rows


def _run_dir(output_root: Path, mutation: str, fraction: float, seed: int) -> Path:
    tag = _format_fraction_tag(fraction)
    if fraction == 1.0:
        return output_root / f"felp_GLY_to_{mutation}_{tag}"
    return output_root / f"felp_GLY_to_{mutation}_{tag}_seed_{seed}"


def _quote_args(args: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in args)


def _generate(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    script_path = Path(args.script_out)
    fractions = tuple(float(item) for item in args.fractions)
    script_path.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"MUTATION={shlex.quote(args.mutation)}",
        f"PYTHON={shlex.quote(_python_felp_parts()[0])}",
        f"FELP_SCRIPT={shlex.quote(_python_felp_parts()[1])}",
        "",
    ]

    for fraction in fractions:
        run_dir = _run_dir(output_root, args.mutation, fraction, args.seed)
        run_args = [
            "run",
            "--topology-file",
            args.topology_file,
            "--target-residue-number",
            str(args.target_residue_number),
            "--target-residue-name",
            "__MUTATION__",
            "--original-residue-name",
            args.original_residue_name,
            "--output-dir",
            str(run_dir),
            "--prior-probability-mode",
            args.prior_probability_mode,
            "--phi-bins",
            str(args.phi_bins),
            "--psi-bins",
            str(args.psi_bins),
            "--progress",
            args.progress,
            "--workers",
            str(args.workers),
            "--chunk-size",
            str(args.chunk_size),
            "--restraint-force-constant",
            str(args.restraint_force_constant),
            "--microstate-sample-fraction",
            str(fraction),
            "--microstate-min-samples",
            str(args.min_samples),
            "--microstate-sampling-seed",
            str(args.seed),
        ]
        for trajectory in args.trajectory:
            run_args.extend(["--trajectory", trajectory])
        if args.max_samples is not None:
            run_args.extend(["--microstate-max-samples", str(args.max_samples)])
        if args.cpu:
            run_args.append("--cpu")
        if args.mute_pyrosetta:
            run_args.append("--mute-pyrosetta")
        if args.overwrite:
            run_args.append("--overwrite")
        lines.extend(
            [
                f"echo 'Running {args.mutation} sampling fraction {fraction:g}'",
                '"${PYTHON}" "${FELP_SCRIPT}" '
                + _quote_args(run_args).replace("__MUTATION__", '"${MUTATION}"'),
                "",
            ]
        )

    analyze_args = [
        "analyze",
        "--output-root",
        str(output_root),
        "--mutation",
        args.mutation,
        "--seed",
        str(args.seed),
        "--analysis-dir",
        str(args.analysis_dir),
    ]
    analyze_args.extend(["--fractions", *(str(fraction) for fraction in fractions)])
    lines.extend(
        [
            "echo 'Analyzing sampling benchmark'",
            f"{shlex.quote(sys.executable)} {shlex.quote(str(Path(__file__).resolve()))} "
            + _quote_args(analyze_args),
            "",
        ]
    )

    script_path.write_text("\n".join(lines), encoding="utf-8")
    script_path.chmod(0o755)
    print(f"Wrote benchmark script: {script_path}")
    return 0


def _probability_l1_distance(
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
) -> float:
    state_ids = set(baseline) | set(candidate)
    return float(
        sum(
            abs(
                float(baseline.get(state_id, {}).get("target_probability", 0.0))
                - float(candidate.get(state_id, {}).get("target_probability", 0.0))
            )
            for state_id in state_ids
        )
    )


def _finite_abs(value: float) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return abs(float(value))


def _max_free_energy_error(
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
) -> float:
    errors: list[float] = []
    for state_id, base_row in baseline.items():
        candidate_row = candidate.get(state_id)
        if candidate_row is None:
            continue
        base_fe = _finite_abs(float(base_row["free_energy_kj_mol"]))
        candidate_fe = _finite_abs(float(candidate_row["free_energy_kj_mol"]))
        if base_fe is None or candidate_fe is None:
            continue
        errors.append(abs(candidate_fe - base_fe))
    return max(errors) if errors else math.nan


def _weighted_probability_error_by_low_ess(
    rows: list[dict[str, Any]],
    threshold: float,
) -> float:
    return float(
        sum(
            float(row["target_probability"])
            for row in rows
            if float(row["effective_sample_size"]) < threshold
        )
    )


def _summarize_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = _load_json(run_dir / "run_summary.json")
    microstates = _load_microstate_summary(run_dir / "microstate_summary.csv")
    return summary, microstates


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_metrics(analysis_dir: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    plot_rows = sorted(rows, key=lambda item: float(item["sample_fraction"]))
    fractions = [float(row["sample_fraction"]) for row in plot_rows]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)
    axes = axes.ravel()
    panels = [
        ("target_probability_l1_error", "Target probability L1 error"),
        ("max_free_energy_abs_error_kj_mol", "Max abs relative F error (kJ/mol)"),
        ("global_effective_sample_fraction", "Global ESS fraction"),
        ("target_probability_low_ess_mass_ess_lt_10", "Target mass with ESS < 10"),
    ]
    for ax, (key, ylabel) in zip(axes, panels, strict=True):
        values = [float(row[key]) for row in plot_rows]
        ax.plot(fractions, values, marker="o", linewidth=1.8)
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("Microstate sample fraction")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    fig.suptitle("FELP sampling-density convergence", fontsize=12)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(analysis_dir / f"sampling_density_convergence.{suffix}", dpi=300)
    plt.close(fig)


def _analyze(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    analysis_dir = Path(args.analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    fractions = tuple(float(item) for item in args.fractions)

    baseline_dir = _run_dir(output_root, args.mutation, 1.0, args.seed)
    baseline_summary, baseline_microstates = _summarize_run(baseline_dir)
    baseline_by_id = {row["state_id"]: row for row in baseline_microstates}

    rows: list[dict[str, Any]] = []
    for fraction in fractions:
        run_dir = _run_dir(output_root, args.mutation, fraction, args.seed)
        summary, microstates = _summarize_run(run_dir)
        by_id = {row["state_id"]: row for row in microstates}
        diagnostics = summary.get("overlap_diagnostics", {})
        low_ess = diagnostics.get("target_probability_low_ess_mass_by_threshold", {})
        sampling = summary.get("sampling") or {}
        rows.append(
            {
                "mutation": args.mutation,
                "sample_fraction": fraction,
                "run_dir": str(run_dir),
                "frame_count": summary.get("frame_count"),
                "reweighted_frame_count": summary.get("reweighted_frame_count"),
                "selected_frame_count": sampling.get("selected_frame_count"),
                "total_assigned_frame_count": sampling.get("total_assigned_frame_count"),
                "overlap_assessment": summary.get("overlap_assessment"),
                "global_effective_sample_size": diagnostics.get(
                    "global_effective_sample_size"
                ),
                "global_effective_sample_fraction": diagnostics.get(
                    "global_effective_sample_fraction"
                ),
                "overlap_coefficient": diagnostics.get("overlap_coefficient"),
                "max_normalized_weight": diagnostics.get("max_normalized_weight"),
                "target_probability_low_ess_mass_ess_lt_10": low_ess.get("ess_lt_10"),
                "target_probability_low_ess_mass_ess_lt_50": low_ess.get("ess_lt_50"),
                "target_probability_l1_error": _probability_l1_distance(
                    baseline_by_id,
                    by_id,
                ),
                "max_free_energy_abs_error_kj_mol": _max_free_energy_error(
                    baseline_by_id,
                    by_id,
                ),
                "target_mass_with_state_ess_lt_10": _weighted_probability_error_by_low_ess(
                    microstates,
                    10.0,
                ),
                "target_mass_with_state_ess_lt_50": _weighted_probability_error_by_low_ess(
                    microstates,
                    50.0,
                ),
            }
        )

    _write_csv(analysis_dir / "sampling_density_metrics.csv", rows)
    _plot_metrics(analysis_dir, rows)
    print(f"Wrote metrics and figures to: {analysis_dir}")
    return 0


def _add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mutation", default="ALA")
    parser.add_argument("--topology-file", required=True)
    parser.add_argument("--trajectory", action="append", required=True)
    parser.add_argument("--target-residue-number", type=int, default=2)
    parser.add_argument("--original-residue-name", default="GLY")
    parser.add_argument("--output-root", default="data/GLY/sampling_benchmark")
    parser.add_argument("--analysis-dir", default="data/GLY/sampling_benchmark/analysis")
    parser.add_argument("--prior-probability-mode", default="raw_counts")
    parser.add_argument("--phi-bins", type=int, default=144)
    parser.add_argument("--psi-bins", type=int, default=144)
    parser.add_argument("--workers", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--restraint-force-constant", type=float, default=1000000.0)
    parser.add_argument("--progress", default="auto")
    parser.add_argument("--fractions", nargs="+", type=float, default=DEFAULT_FRACTIONS)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", default=True)
    parser.add_argument("--mute-pyrosetta", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument(
        "--script-out",
        default="data/GLY/sampling_benchmark/run_sampling_benchmark.sh",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and analyze FELP sampling-density benchmarks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser(
        "generate",
        help="Write a runnable shell script for baseline and sampling sweeps.",
    )
    _add_common_run_args(generate_parser)

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze completed benchmark run directories.",
    )
    analyze_parser.add_argument("--output-root", required=True)
    analyze_parser.add_argument("--analysis-dir", required=True)
    analyze_parser.add_argument("--mutation", required=True)
    analyze_parser.add_argument("--seed", type=int, default=0)
    analyze_parser.add_argument("--fractions", nargs="+", type=float, default=DEFAULT_FRACTIONS)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate":
        return _generate(args)
    if args.command == "analyze":
        return _analyze(args)
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
