"""Application entry points for FELP."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
import os
import sys
import time
from typing import Any, Sequence

from felp.cli import (
    build_felp_plot_config,
    build_microstate_config,
    build_msm_fit_config,
    build_msm_plot_config,
    build_msm_sample_export_config,
    build_parser,
    build_reweighting_qc_config,
    build_run_config,
)
from felp.core import OutputConfig
from felp.io import ResultsWriter, TrajectoryLoader, load_frame_results_csv
from felp.msm import PyEMMAPipeline, export_state_samples, load_bundle
from felp.parallel import FrameExecutor
from felp.plotting import generate_felp_plots, generate_msm_plots
from felp.protocol_summary import build_protocol_summary
from felp.reweighting import (
    MicrostateAtlas,
    PrecomputedStateAssignmentPolicy,
    ReweightingAggregator,
    SamplingManifest,
    ensure_existing_results_match_manifest,
    is_reweighting_eligible_frame,
    load_or_create_sampling_manifest,
    manifest_counts_for_atlas,
    selected_jobs_from_manifest,
    write_auto_microstate_file,
)

_OVERLAP_SCOPE_NOTE = (
    "These diagnostics only assess overlap on the sampled reference support and "
    "the chosen microstate partition."
)
_OVERLAP_LIMITATIONS_NOTE = (
    "They cannot prove the absence of unseen target-only states outside the "
    "sampled reference support."
)
_OVERLAP_VALIDATION_NOTE = (
    "When both directions are available, compare forward and reverse FELP runs "
    "as a stronger overlap validation."
)
_OVERLAP_ASSESSMENT_THRESHOLDS = {
    "poor": {
        "global_effective_sample_fraction_lt": 0.01,
        "target_probability_low_ess_mass_ess_lt_10_gt": 0.25,
    },
    "caution": {
        "global_effective_sample_fraction_lt": 0.05,
        "target_probability_low_ess_mass_ess_lt_10_gt": 0.10,
        "overlap_coefficient_lt": 0.20,
    },
}


def _write_stdout(payload: dict) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _log_phase(message: str) -> None:
    print(f"[FELP] {message}", file=sys.stderr, flush=True)


def _build_execution_context(config):
    """Build the atlas, assignment policy, and optional bundle for a FELP run."""

    atlas = MicrostateAtlas.from_config(config.microstates)
    if config.microstates.source == "msm_bundle":
        bundle = load_bundle(config.microstates.msm_bundle_dir)
        bundle.validate_against_input(config.input)
        return atlas, PrecomputedStateAssignmentPolicy(), bundle
    return atlas, None, None


def _materialize_auto_microstates(config, jobs):
    """Write auto-generated CV microstates and return a file-backed config."""

    if config.microstates.source != "auto":
        return config, None

    path = os.path.join(config.output.output_dir, "auto_microstates.json")
    auto_file = write_auto_microstate_file(
        jobs=jobs,
        config=config,
        output_path=path,
    )
    microstates = replace(
        config.microstates,
        source="file",
        definition_file=auto_file.path,
        reference_probability_mode="raw_counts",
    )
    return replace(config, microstates=microstates), auto_file


def _shared_microstate_metadata(microstate_results) -> dict[str, Any]:
    """Return the shared metadata written onto every microstate result row."""

    if not microstate_results:
        return {}
    return dict(microstate_results[0].metadata)


def _build_overlap_diagnostics(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return a structured overlap-diagnostic bundle for JSON summaries."""

    if not metadata:
        return {}
    return {
        "eligible_frame_count": int(metadata.get("eligible_frame_count", 0)),
        "global_effective_sample_size": float(
            metadata.get("global_effective_sample_size", 0.0)
        ),
        "global_effective_sample_fraction": float(
            metadata.get("global_effective_sample_fraction", 0.0)
        ),
        "max_normalized_weight": float(metadata.get("max_normalized_weight", 0.0)),
        "top_weight_mass_by_fraction": {
            "top_1_percent": float(metadata.get("top_1_percent_weight_mass", 0.0)),
            "top_5_percent": float(metadata.get("top_5_percent_weight_mass", 0.0)),
            "top_10_percent": float(metadata.get("top_10_percent_weight_mass", 0.0)),
        },
        "overlap_coefficient": float(metadata.get("overlap_coefficient", 0.0)),
        "target_probability_low_ess_mass_by_threshold": {
            "ess_lt_10": float(
                metadata.get("target_probability_low_ess_mass_ess_lt_10", 0.0)
            ),
            "ess_lt_50": float(
                metadata.get("target_probability_low_ess_mass_ess_lt_50", 0.0)
            ),
        },
        "assessment_thresholds": _OVERLAP_ASSESSMENT_THRESHOLDS,
        "scope": _OVERLAP_SCOPE_NOTE,
        "limitations": _OVERLAP_LIMITATIONS_NOTE,
        "recommended_validation": _OVERLAP_VALIDATION_NOTE,
    }


def _build_reweighting_qc_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return the shared by-state reweighting QC summary."""

    qc_summary = metadata.get("reweighting_qc")
    return dict(qc_summary) if isinstance(qc_summary, dict) else {}


def _format_overlap_metric(value: float) -> str:
    """Return a compact numeric string for advisory messages."""

    return f"{float(value):.4g}"


def _assess_overlap_diagnostics(
    overlap_diagnostics: dict[str, Any],
) -> tuple[str | None, list[str]]:
    """Assign an advisory trust label to the overlap-diagnostic bundle."""

    if not overlap_diagnostics:
        return None, []

    ess_fraction = float(overlap_diagnostics["global_effective_sample_fraction"])
    overlap_coefficient = float(overlap_diagnostics["overlap_coefficient"])
    low_ess_mass_10 = float(
        overlap_diagnostics["target_probability_low_ess_mass_by_threshold"]["ess_lt_10"]
    )

    poor_reasons: list[str] = []
    caution_reasons: list[str] = []

    poor_thresholds = _OVERLAP_ASSESSMENT_THRESHOLDS["poor"]
    caution_thresholds = _OVERLAP_ASSESSMENT_THRESHOLDS["caution"]

    if ess_fraction < poor_thresholds["global_effective_sample_fraction_lt"]:
        poor_reasons.append(
            "global ESS fraction is below the advisory poor threshold "
            f"({_format_overlap_metric(ess_fraction)} < "
            f"{poor_thresholds['global_effective_sample_fraction_lt']})."
        )
    elif ess_fraction < caution_thresholds["global_effective_sample_fraction_lt"]:
        caution_reasons.append(
            "global ESS fraction is below the advisory caution threshold "
            f"({_format_overlap_metric(ess_fraction)} < "
            f"{caution_thresholds['global_effective_sample_fraction_lt']})."
        )

    if low_ess_mass_10 > poor_thresholds["target_probability_low_ess_mass_ess_lt_10_gt"]:
        poor_reasons.append(
            "target probability mass in states with ESS < 10 exceeds the advisory "
            "poor threshold "
            f"({_format_overlap_metric(low_ess_mass_10)} > "
            f"{poor_thresholds['target_probability_low_ess_mass_ess_lt_10_gt']})."
        )
    elif (
        low_ess_mass_10
        > caution_thresholds["target_probability_low_ess_mass_ess_lt_10_gt"]
    ):
        caution_reasons.append(
            "target probability mass in states with ESS < 10 exceeds the advisory "
            "caution threshold "
            f"({_format_overlap_metric(low_ess_mass_10)} > "
            f"{caution_thresholds['target_probability_low_ess_mass_ess_lt_10_gt']})."
        )

    if overlap_coefficient < caution_thresholds["overlap_coefficient_lt"]:
        caution_reasons.append(
            "discrete overlap coefficient is below the advisory caution threshold "
            f"({_format_overlap_metric(overlap_coefficient)} < "
            f"{caution_thresholds['overlap_coefficient_lt']})."
        )

    if poor_reasons:
        return "poor", [*poor_reasons, *caution_reasons]
    if caution_reasons:
        return "caution", caution_reasons
    return "ok", []


def _build_reweighting_summary(
    *,
    frame_count: int,
    reweighted_frame_count: int,
    microstate_results,
    output_dir: str,
) -> dict[str, Any]:
    """Build the shared JSON summary fields for FELP reweighting outputs."""

    metadata = _shared_microstate_metadata(microstate_results)
    overlap_diagnostics = _build_overlap_diagnostics(metadata)
    reweighting_qc = _build_reweighting_qc_summary(metadata)
    overlap_assessment, overlap_assessment_reasons = _assess_overlap_diagnostics(
        overlap_diagnostics
    )
    return {
        "frame_count": frame_count,
        "reweighted_frame_count": reweighted_frame_count,
        "microstate_count": len(microstate_results),
        "reference_probability_source": metadata.get("reference_probability_source"),
        "prior_probability_mode": metadata.get("prior_probability_mode"),
        "reference_count_scope": metadata.get("reference_count_scope"),
        "reference_probability_total": float(
            sum(result.reference_probability for result in microstate_results)
        ),
        "target_probability_total": float(
            sum(result.target_probability for result in microstate_results)
        ),
        "unsupported_state_count": int(metadata.get("unsupported_state_count", 0)),
        "unsupported_state_ids": list(metadata.get("unsupported_state_ids", [])),
        "sampling": metadata.get("sampling"),
        "overlap_assessment": overlap_assessment,
        "overlap_assessment_reasons": overlap_assessment_reasons,
        "overlap_diagnostics": overlap_diagnostics,
        "reweighting_qc": reweighting_qc,
        "output_dir": output_dir,
    }


def run_pipeline(config) -> dict:
    """Execute the main FELP perturbation and reweighting workflow."""
    pipeline_start = time.perf_counter()
    writer = ResultsWriter(config.output)
    writer.prepare_output_dir()

    bundle = None
    if config.microstates.source == "msm_bundle":
        bundle = load_bundle(config.microstates.msm_bundle_dir)
        bundle.validate_against_input(config.input)
    load_start = time.perf_counter()
    _log_phase("loading trajectory frames")
    jobs = TrajectoryLoader(
        config.input,
        bundle=bundle,
        state_level=config.microstates.state_level,
        load_chunk_size=max(1024, int(config.parallel.chunk_size)),
    ).load_jobs()
    _log_phase(
        f"loaded {len(jobs)} frame jobs in {time.perf_counter() - load_start:.1f}s"
    )

    auto_start = time.perf_counter()
    config, auto_microstate_file = _materialize_auto_microstates(config, jobs)
    if auto_microstate_file is not None:
        _log_phase(
            "auto microstates: "
            f"{auto_microstate_file.state_count} states from "
            f"{auto_microstate_file.frame_count} frames in "
            f"{time.perf_counter() - auto_start:.1f}s"
        )
    writer.write_run_config(config)
    context_start = time.perf_counter()
    atlas, assignment_policy, bundle = _build_execution_context(config)
    _log_phase(f"built execution context in {time.perf_counter() - context_start:.1f}s")
    manifest_path = os.path.join(
        config.output.output_dir,
        config.sampling.sampling_manifest_name,
    )
    sampling_start = time.perf_counter()
    _log_phase(
        f"building sampling manifest for {len(jobs)} frames "
        "(assign each frame to a microstate; may take a few minutes at large N)"
    )
    sampling_manifest = load_or_create_sampling_manifest(
        manifest_path=manifest_path,
        jobs=jobs,
        atlas=atlas,
        config=config,
    )
    sampled_jobs = selected_jobs_from_manifest(jobs, sampling_manifest)
    _log_phase(
        f"selected {len(sampled_jobs)} sampled jobs in "
        f"{time.perf_counter() - sampling_start:.1f}s"
    )

    existing_results = writer.read_frame_results() if config.output.resume else []
    ensure_existing_results_match_manifest(
        (result.frame_index for result in existing_results),
        sampling_manifest,
    )
    completed_indices = {result.frame_index for result in existing_results}
    pending_jobs = [
        job for job in sampled_jobs if job.frame_index not in completed_indices
    ]

    _log_phase(
        f"processing {len(pending_jobs)} pending frames with "
        f"{config.parallel.workers} workers and chunk size {config.parallel.chunk_size}"
    )
    frame_start = time.perf_counter()
    frame_results = FrameExecutor(
        config,
        atlas,
        assignment_policy=assignment_policy,
    ).run(pending_jobs)
    _log_phase(f"frame processing finished in {time.perf_counter() - frame_start:.1f}s")
    all_frame_results = sorted(
        [*existing_results, *frame_results],
        key=lambda item: item.frame_index,
    )

    aggregator = ReweightingAggregator(
        config.energy.temperature_kelvin,
        qc_config=config.reweighting_qc,
    )
    sampling_metadata = sampling_manifest.metadata()
    reference_counts = (
        manifest_counts_for_atlas(sampling_manifest, atlas)
        if not config.sampling.is_noop
        else None
    )
    microstate_results = aggregator.aggregate(
        all_frame_results,
        atlas,
        reference_counts=reference_counts,
        sampling_metadata=sampling_metadata,
    )

    writer.write_frame_results(all_frame_results)
    writer.write_microstate_results(microstate_results)

    failed_frame_count = sum(
        1 for result in all_frame_results if result.metadata.get("status") == "failed"
    )
    new_failed_frame_count = sum(
        1 for result in frame_results if result.metadata.get("status") == "failed"
    )
    reweighted_frame_count = sum(
        1 for result in all_frame_results if is_reweighting_eligible_frame(result)
    )
    histogram_frames = [
        result
        for result in all_frame_results
        if result.metadata.get("status") != "failed"
    ]
    empirical_histogram = Counter(result.microstate_label for result in histogram_frames)
    summary = _build_reweighting_summary(
        frame_count=len(all_frame_results),
        reweighted_frame_count=reweighted_frame_count,
        microstate_results=microstate_results,
        output_dir=config.output.output_dir,
    )
    summary.update(
        {
            "new_frame_count": len(frame_results),
            "failed_frame_count": failed_frame_count,
            "new_failed_frame_count": new_failed_frame_count,
            "microstate_histogram": dict(empirical_histogram),
            "microstate_histogram_kind": "empirical_frame_counts",
            "sampling": sampling_metadata,
            "microstate_assigned_histogram": dict(sampling_manifest.assigned_counts),
            "microstate_selected_histogram": dict(sampling_manifest.selected_counts),
            "state_reweighting_estimator": (
                "pi_target(i) proportional to "
                "pi_reference(i) * <exp(-beta * delta_u)>_i"
            ),
            "microstate_source": config.microstates.source,
            "microstate_definition_state_level": config.microstates.state_level,
            "msm_bundle_dir": config.microstates.msm_bundle_dir,
            "configured_cvs": [spec.name for spec in config.restraints.cv_specs],
            "configured_distance_cvs": [
                spec.name for spec in config.restraints.distance_cv_specs
            ],
            "analysis_cv_names": sorted(
                {
                    cv_name
                    for definition in atlas.definitions
                    for cv_name in definition.cv_ranges
                }
            ),
            "auto_microstate_file": (
                None if auto_microstate_file is None else auto_microstate_file.path
            ),
            "auto_microstate_metadata": (
                None if auto_microstate_file is None else auto_microstate_file.metadata
            ),
            "protocol": build_protocol_summary(config),
        }
    )
    writer.write_summary(summary)
    _log_phase(f"pipeline finished in {time.perf_counter() - pipeline_start:.1f}s")
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


def _repartition_frame_results(
    frame_results: list,
    atlas: "MicrostateAtlas",
) -> list:
    """Re-assign every frame to a new atlas using stored CV measurements."""

    from dataclasses import replace as _replace

    repartitioned: list = []
    unassigned_count = 0
    for result in frame_results:
        metadata = getattr(result, "metadata", {}) or {}
        measurements = metadata.get("assignment_measurements")
        if measurements is None:
            if result.phi_degrees is not None and result.psi_degrees is not None:
                measurements = {"phi": result.phi_degrees, "psi": result.psi_degrees}
            else:
                repartitioned.append(result)
                continue

        try:
            assignment = atlas.assign(measurements)
        except Exception:
            unassigned_count += 1
            repartitioned.append(result)
            continue

        updated_metadata = dict(metadata)
        updated_metadata["phi_center"] = assignment.phi_center
        updated_metadata["psi_center"] = assignment.psi_center
        updated_metadata["plot_x"] = assignment.plot_x
        updated_metadata["plot_y"] = assignment.plot_y
        repartitioned.append(
            _replace(
                result,
                microstate_id=assignment.state_id,
                microstate_label=assignment.label,
                metadata=updated_metadata,
            )
        )

    if unassigned_count > 0:
        print(
            f"[FELP] repartition: {unassigned_count} frame(s) could not be assigned "
            f"to the new atlas",
            file=sys.stderr,
        )
    return repartitioned


def _build_auto_atlas_from_frame_results(
    frame_results: list,
    microstate_config,
    output_dir: str,
) -> "MicrostateAtlas":
    """Build an auto-generated atlas directly from stored CV measurements."""

    import numpy as np
    from felp.reweighting import build_auto_microstate_payload

    cv_arrays: dict[str, list[float]] = {}
    for result in frame_results:
        metadata = getattr(result, "metadata", {}) or {}
        measurements = metadata.get("assignment_measurements")
        if measurements is None:
            if result.phi_degrees is not None and result.psi_degrees is not None:
                measurements = {"phi": result.phi_degrees, "psi": result.psi_degrees}
            else:
                continue
        for name, value in measurements.items():
            cv_arrays.setdefault(name, []).append(float(value))

    cv_names = tuple(cv_arrays.keys())
    if len(cv_names) != 2:
        from felp.core import ConfigurationError
        raise ConfigurationError(
            f"Repartition with auto microstates requires exactly 2 CVs in the stored "
            f"frame results, but found {len(cv_names)}: {', '.join(cv_names) or '<none>'}."
        )

    cv_values = {name: np.asarray(values, dtype=float) for name, values in cv_arrays.items()}
    payload = build_auto_microstate_payload(
        cv_values=cv_values,
        cv_names=cv_names,
        target_frames_per_state=microstate_config.auto_target_frames_per_state,
        min_bins_per_axis=microstate_config.auto_min_bins_per_axis,
        max_bins_per_axis=microstate_config.auto_max_bins_per_axis,
        max_total_states=microstate_config.auto_max_total_states,
    )

    auto_path = os.path.join(output_dir, "auto_microstates.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(auto_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    state_count = len(payload["states"])
    print(
        f"[FELP] repartition: auto-generated {state_count} microstates from "
        f"{len(cv_values[cv_names[0]])} frames -> {auto_path}",
        file=sys.stderr,
    )

    from felp.core import MicrostateConfig
    from dataclasses import replace as _replace

    file_config = _replace(
        microstate_config,
        source="file",
        definition_file=auto_path,
    )
    return MicrostateAtlas.from_config(file_config)


def summarize_results(namespace) -> dict:
    """Recompute microstate reweighting from an existing frame-results CSV."""

    microstate_config = build_microstate_config(namespace)
    repartition = getattr(namespace, "repartition", False)
    overwrite = getattr(namespace, "overwrite", False)

    writer = ResultsWriter(
        OutputConfig(
            output_dir=namespace.output_dir,
            resume=not overwrite,
            overwrite=overwrite,
        )
    )
    writer.prepare_output_dir()

    frame_results = load_frame_results_csv(namespace.raw_frame_results)

    if repartition:
        _log_phase("repartitioning frames to new microstate definitions")
        if microstate_config.source == "auto":
            atlas = _build_auto_atlas_from_frame_results(
                frame_results, microstate_config, namespace.output_dir
            )
        else:
            atlas = MicrostateAtlas.from_config(microstate_config)
        frame_results = _repartition_frame_results(frame_results, atlas)
        print(
            f"[FELP] repartitioned {len(frame_results)} frames into "
            f"{len(atlas.definitions)} microstates",
            file=sys.stderr,
        )
    else:
        atlas = MicrostateAtlas.from_config(microstate_config)

    sampling_manifest = None
    sampling_manifest_path = os.path.join(
        os.path.dirname(namespace.raw_frame_results),
        "sampling_manifest.json",
    )
    if os.path.isfile(sampling_manifest_path):
        with open(sampling_manifest_path, encoding="utf-8") as handle:
            sampling_manifest = SamplingManifest.from_dict(json.load(handle))
    reference_counts = (
        manifest_counts_for_atlas(sampling_manifest, atlas)
        if sampling_manifest is not None and sampling_manifest.skipped_frame_count > 0
        else None
    )
    sampling_metadata = (
        sampling_manifest.metadata() if sampling_manifest is not None else None
    )

    microstate_results = ReweightingAggregator(
        namespace.temperature_kelvin,
        qc_config=build_reweighting_qc_config(namespace),
    ).aggregate(
        frame_results,
        atlas,
        reference_counts=reference_counts,
        sampling_metadata=sampling_metadata,
    )
    writer.write_microstate_results(microstate_results)
    summary = _build_reweighting_summary(
        frame_count=len(frame_results),
        reweighted_frame_count=sum(
            1 for result in frame_results if is_reweighting_eligible_frame(result)
        ),
        microstate_results=microstate_results,
        output_dir=namespace.output_dir,
    )
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
