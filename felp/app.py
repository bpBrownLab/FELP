"""Application entry points for FELP."""

from __future__ import annotations

from collections import Counter
import csv
import json
from typing import Sequence

from felp.cli import (
    build_felp_plot_config,
    build_microstate_config,
    build_msm_fit_config,
    build_msm_plot_config,
    build_msm_sample_export_config,
    build_parser,
    build_run_config,
)
from felp.core import FrameResult, OutputConfig
from felp.io import ResultsWriter, TrajectoryLoader
from felp.msm import PyEMMAPipeline, export_state_samples, load_bundle
from felp.parallel import FrameExecutor
from felp.plotting import generate_felp_plots, generate_msm_plots
from felp.reweighting import (
    MicrostateAtlas,
    PrecomputedStateAssignmentPolicy,
    ReweightingAggregator,
)


def _parse_optional_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _parse_optional_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _write_stdout(payload: dict) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _load_frame_results_csv(path: str) -> list[FrameResult]:
    """Read frame-level FELP results from an arbitrary CSV path."""

    frame_results: list[FrameResult] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            frame_results.append(
                FrameResult(
                    frame_index=int(row["frame_index"]),
                    stride_number=int(row["stride_number"]),
                    trajectory_id=_parse_optional_int(row.get("trajectory_id")),
                    source_frame_index=_parse_optional_int(row.get("source_frame_index")),
                    microstate_id=row["microstate_id"],
                    microstate_label=row["microstate_label"],
                    phi_degrees=_parse_optional_float(row.get("phi_degrees")),
                    psi_degrees=_parse_optional_float(row.get("psi_degrees")),
                    reference_energy_kj_mol=float(row["reference_energy_kj_mol"]),
                    perturbed_energy_kj_mol=float(row["perturbed_energy_kj_mol"]),
                    delta_u_kj_mol=float(row["delta_u_kj_mol"]),
                    reference_relaxed_phi_degrees=_parse_optional_float(
                        row.get("reference_relaxed_phi_degrees")
                    ),
                    reference_relaxed_psi_degrees=_parse_optional_float(
                        row.get("reference_relaxed_psi_degrees")
                    ),
                    perturbed_relaxed_phi_degrees=_parse_optional_float(
                        row.get("perturbed_relaxed_phi_degrees")
                    ),
                    perturbed_relaxed_psi_degrees=_parse_optional_float(
                        row.get("perturbed_relaxed_psi_degrees")
                    ),
                    assignment_source=row.get("assignment_source", "phi_psi"),
                    state_level=row.get("state_level", "microstate"),
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                )
            )
    return frame_results


def _build_execution_context(config):
    """Build the atlas, assignment policy, and optional bundle for a FELP run."""

    atlas = MicrostateAtlas.from_config(config.microstates)
    if config.microstates.source == "msm_bundle":
        bundle = load_bundle(config.microstates.msm_bundle_dir)
        bundle.validate_against_input(config.input)
        return atlas, PrecomputedStateAssignmentPolicy(), bundle
    return atlas, None, None


def run_pipeline(config) -> dict:
    """Execute the main FELP perturbation and reweighting workflow."""
    writer = ResultsWriter(config.output)
    writer.prepare_output_dir()
    writer.write_run_config(config)

    atlas, assignment_policy, bundle = _build_execution_context(config)
    jobs = TrajectoryLoader(
        config.input,
        bundle=bundle,
        state_level=config.microstates.state_level,
    ).load_jobs()

    existing_results = writer.read_frame_results() if config.output.resume else []
    completed_indices = {result.frame_index for result in existing_results}
    pending_jobs = [job for job in jobs if job.frame_index not in completed_indices]

    frame_results = FrameExecutor(
        config,
        atlas,
        assignment_policy=assignment_policy,
    ).run(pending_jobs)
    all_frame_results = sorted(
        [*existing_results, *frame_results],
        key=lambda item: item.frame_index,
    )

    aggregator = ReweightingAggregator(config.energy.temperature_kelvin)
    microstate_results = aggregator.aggregate(all_frame_results, atlas)

    writer.write_frame_results(all_frame_results)
    writer.write_microstate_results(microstate_results)

    summary = {
        "frame_count": len(all_frame_results),
        "new_frame_count": len(frame_results),
        "microstate_count": len(microstate_results),
        "microstate_histogram": dict(
            Counter(result.microstate_label for result in all_frame_results)
        ),
        "microstate_source": config.microstates.source,
        "state_level": config.microstates.state_level,
        "msm_bundle_dir": config.microstates.msm_bundle_dir,
        "configured_cvs": [spec.name for spec in config.restraints.cv_specs],
        "output_dir": config.output.output_dir,
    }
    writer.write_summary(summary)
    return summary


def validate_microstates(namespace) -> dict:
    """Load and summarize microstates without running the full pipeline."""
    config = build_microstate_config(namespace)
    atlas = MicrostateAtlas.from_config(config)
    summary = atlas.summary()
    if config.source == "msm_bundle":
        bundle = load_bundle(config.msm_bundle_dir)
        summary["bundle"] = bundle.summary()
    _write_stdout(summary)
    return summary


def summarize_results(namespace) -> dict:
    """Recompute microstate reweighting from an existing frame-results CSV."""
    atlas = MicrostateAtlas.from_config(build_microstate_config(namespace))
    writer = ResultsWriter(
        OutputConfig(
            output_dir=namespace.output_dir,
            resume=True,
        )
    )
    writer.prepare_output_dir()

    frame_results = _load_frame_results_csv(namespace.raw_frame_results)

    microstate_results = ReweightingAggregator(namespace.temperature_kelvin).aggregate(
        frame_results,
        atlas,
    )
    writer.write_microstate_results(microstate_results)
    summary = {
        "frame_count": len(frame_results),
        "microstate_count": len(microstate_results),
        "output_dir": namespace.output_dir,
    }
    writer.write_summary(summary)
    _write_stdout(summary)
    return summary


def fit_msm_pipeline(config) -> dict:
    """Fit a PyEMMA model and export an FELP-readable MSM bundle."""

    result = PyEMMAPipeline(config).run()
    summary = result.bundle.summary()
    summary.update(
        {
            "output_dir": config.output_dir,
            "feature_spec_file": config.feature_spec_file,
            "frame_count": int(sum(len(values) for values in result.cluster_dtrajs)),
        }
    )
    _write_stdout(summary)
    return summary


def inspect_msm_bundle(namespace) -> dict:
    """Inspect a saved MSM bundle and print a concise summary."""

    summary = load_bundle(namespace.msm_bundle_dir).summary()
    _write_stdout(summary)
    return summary


def export_msm_bundle_samples(namespace) -> dict:
    """Export representative structures for MSM bundle states."""

    summary = export_state_samples(build_msm_sample_export_config(namespace))
    _write_stdout(summary)
    return summary


def plot_msm_bundle(namespace) -> dict:
    """Generate publication-style plots for an MSM bundle."""

    summary = generate_msm_plots(build_msm_plot_config(namespace))
    _write_stdout(summary)
    return summary


def plot_felp_outputs(namespace) -> dict:
    """Generate publication-style plots for FELP outputs."""

    summary = generate_felp_plots(build_felp_plot_config(namespace))
    _write_stdout(summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the FELP application."""
    parser = build_parser()
    namespace = parser.parse_args(argv)
    if namespace.command == "run":
        summary = run_pipeline(build_run_config(namespace))
        _write_stdout(summary)
        return 0
    if namespace.command == "validate-microstates":
        validate_microstates(namespace)
        return 0
    if namespace.command == "summarize":
        summarize_results(namespace)
        return 0
    if namespace.command == "msm-fit":
        fit_msm_pipeline(build_msm_fit_config(namespace))
        return 0
    if namespace.command == "msm-inspect":
        inspect_msm_bundle(namespace)
        return 0
    if namespace.command == "msm-export-samples":
        export_msm_bundle_samples(namespace)
        return 0
    if namespace.command == "msm-plot":
        plot_msm_bundle(namespace)
        return 0
    if namespace.command == "felp-plot":
        plot_felp_outputs(namespace)
        return 0
    parser.error(f"Unknown command: {namespace.command}")
    return 2
