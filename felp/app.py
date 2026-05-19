"""Application entry points for FELP."""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from dataclasses import replace
import json
import math
import os
import sys
import time
from typing import Any, Sequence

from felp.cli import (
    build_ess_compare_config,
    build_felp_plot_config,
    build_md_kl_plot_config,
    build_microstate_config,
    build_msm_fit_config,
    build_msm_plot_config,
    build_msm_sample_export_config,
    build_parser,
    build_reweighting_config,
    build_reweighting_qc_config,
    build_run_config,
)
from felp.core import ConfigurationError, OutputConfig
from felp.io import ResultsWriter, TrajectoryLoader, load_frame_results_csv
from felp.msm import PyEMMAPipeline, export_state_samples, load_bundle
from felp.parallel import FrameExecutor
from felp.plotting import (
    generate_ess_comparison_plots,
    generate_felp_plots,
    generate_md_kl_plot,
    generate_msm_plots,
)
from felp.protocol_summary import build_protocol_summary
from felp.reweighting import (
    MicrostateAtlas,
    PrecomputedStateAssignmentPolicy,
    ReweightingOrchestrator,
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


def _execute_reweighting(
    *,
    config,
    frame_results,
    atlas,
    reference_counts,
    sampling_metadata,
    writer: ResultsWriter,
):
    """Run configured reweighting backend(s) and write method outputs."""

    orchestrator = ReweightingOrchestrator(
        temperature_kelvin=config.energy.temperature_kelvin,
        config=config.reweighting,
        qc_config=config.reweighting_qc,
    )
    methods = config.reweighting.methods
    if len(methods) == 1:
        outcome = orchestrator.run_single(
            frame_results,
            atlas,
            reference_counts=reference_counts,
            sampling_metadata=sampling_metadata,
        )
        writer.write_microstate_results(outcome.microstate_results)
        frame_weights = outcome.auxiliary.get("frame_weights")
        if frame_weights:
            writer.write_frame_weights(frame_weights)
        return outcome.microstate_results, outcome

    outcomes = orchestrator.run(
        frame_results,
        atlas,
        reference_counts=reference_counts,
        sampling_metadata=sampling_metadata,
        output_dir=config.output.output_dir,
        writer=writer,
    )
    primary = outcomes[methods[0]]
    writer.write_microstate_results(primary.microstate_results)
    return primary.microstate_results, primary


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

    sampling_metadata = sampling_manifest.metadata()
    reference_counts = (
        manifest_counts_for_atlas(sampling_manifest, atlas)
        if not config.sampling.is_noop
        else None
    )
    microstate_results, reweighting_outcome = _execute_reweighting(
        config=config,
        frame_results=all_frame_results,
        atlas=atlas,
        reference_counts=reference_counts,
        sampling_metadata=sampling_metadata,
        writer=writer,
    )

    writer.write_frame_results(all_frame_results)

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
            "reweighting_method": reweighting_outcome.method_id,
            "reweighting_methods": list(config.reweighting.methods),
            "state_reweighting_estimator": reweighting_outcome.state_reweighting_estimator,
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


def _stored_assignment_measurements(result) -> dict[str, float] | None:
    """Return stored assignment CVs, with phi/psi columns as a legacy fallback."""

    metadata = getattr(result, "metadata", {}) or {}
    measurements = metadata.get("assignment_measurements")
    if isinstance(measurements, dict):
        return {str(name): float(value) for name, value in measurements.items()}
    if result.phi_degrees is not None and result.psi_degrees is not None:
        return {"phi": float(result.phi_degrees), "psi": float(result.psi_degrees)}
    return None


def _repartition_frame_results(
    frame_results: list,
    atlas: "MicrostateAtlas",
    *,
    grid_bins: tuple[int, int] | None = None,
) -> list:
    """Re-assign every frame to a new atlas using stored CV measurements."""

    if grid_bins is not None:
        fast = _repartition_frame_results_to_phi_psi_grid(
            frame_results,
            atlas,
            phi_bins=int(grid_bins[0]),
            psi_bins=int(grid_bins[1]),
        )
        if fast is not None:
            return fast

    fast = _repartition_frame_results_to_complete_2d_grid(frame_results, atlas)
    if fast is not None:
        return fast

    repartitioned: list = []
    unassigned_count = 0
    for result in frame_results:
        metadata = getattr(result, "metadata", {}) or {}
        measurements = _stored_assignment_measurements(result)
        if measurements is None:
            repartitioned.append(result)
            continue

        try:
            assignment = atlas.assign(measurements)
        except Exception:
            unassigned_count += 1
            repartitioned.append(result)
            continue

        updated_metadata = dict(metadata)
        updated_metadata["assignment_measurements"] = assignment.measurements
        updated_metadata["cv_centers"] = assignment.cv_centers
        if assignment.phi_center is not None:
            updated_metadata["phi_center"] = assignment.phi_center
        if assignment.psi_center is not None:
            updated_metadata["psi_center"] = assignment.psi_center
        updated_metadata["plot_x"] = assignment.plot_x
        updated_metadata["plot_y"] = assignment.plot_y
        repartitioned.append(
            replace(
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


def _merge_edge_value(existing: float | None, value: float) -> float | None:
    """Merge shared grid-edge values while tolerating tiny JSON roundoff."""

    if existing is None:
        return float(value)
    if math.isclose(float(existing), float(value), rel_tol=0.0, abs_tol=1.0e-10):
        return float(existing)
    return None


def _auto_complete_2d_grid_spec(
    atlas: "MicrostateAtlas",
) -> tuple[tuple[str, str], list[float], list[float], dict[tuple[int, int], Any]] | None:
    """Return direct-binning metadata for complete 2D auto grids when available."""

    definitions = tuple(atlas.definitions)
    if not definitions:
        return None

    first_indices = definitions[0].metadata.get("bin_indices")
    if not isinstance(first_indices, dict) or len(first_indices) != 2:
        return None
    cv_names = tuple(str(name) for name in first_indices)
    if len(set(cv_names)) != 2:
        return None
    x_name, y_name = cv_names

    by_index: dict[tuple[int, int], Any] = {}
    x_edges_by_position: dict[int, float] = {}
    y_edges_by_position: dict[int, float] = {}

    for definition in definitions:
        bin_indices = definition.metadata.get("bin_indices")
        if not isinstance(bin_indices, dict) or set(bin_indices) != set(cv_names):
            return None
        if set(definition.cv_ranges) != set(cv_names):
            return None
        try:
            x_index = int(bin_indices[x_name])
            y_index = int(bin_indices[y_name])
            x_lower, x_upper = (
                float(value) for value in definition.cv_ranges[x_name]
            )
            y_lower, y_upper = (
                float(value) for value in definition.cv_ranges[y_name]
            )
        except Exception:
            return None
        if x_index < 0 or y_index < 0 or x_upper < x_lower or y_upper < y_lower:
            return None

        key = (x_index, y_index)
        if key in by_index:
            return None
        by_index[key] = definition

        for position, value, edge_map in (
            (x_index, x_lower, x_edges_by_position),
            (x_index + 1, x_upper, x_edges_by_position),
            (y_index, y_lower, y_edges_by_position),
            (y_index + 1, y_upper, y_edges_by_position),
        ):
            merged = _merge_edge_value(edge_map.get(position), value)
            if merged is None:
                return None
            edge_map[position] = merged

    x_bin_count = max(index[0] for index in by_index) + 1
    y_bin_count = max(index[1] for index in by_index) + 1
    if len(by_index) != x_bin_count * y_bin_count:
        return None
    if set(x_edges_by_position) != set(range(x_bin_count + 1)):
        return None
    if set(y_edges_by_position) != set(range(y_bin_count + 1)):
        return None
    if set(by_index) != {
        (x_index, y_index)
        for x_index in range(x_bin_count)
        for y_index in range(y_bin_count)
    }:
        return None

    x_edges = [x_edges_by_position[index] for index in range(x_bin_count + 1)]
    y_edges = [y_edges_by_position[index] for index in range(y_bin_count + 1)]
    if any(
        upper < lower
        for lower, upper in zip(x_edges[:-1], x_edges[1:], strict=True)
    ):
        return None
    if any(
        upper < lower
        for lower, upper in zip(y_edges[:-1], y_edges[1:], strict=True)
    ):
        return None

    return (x_name, y_name), x_edges, y_edges, by_index


def _edge_bin_index(edges: list[float], value: float) -> int | None:
    """Return the atlas-compatible bin index for a value on monotonic edges."""

    if not math.isfinite(value) or value < edges[0] or value > edges[-1]:
        return None
    return min(max(bisect_left(edges, value) - 1, 0), len(edges) - 2)


def _repartition_frame_results_to_complete_2d_grid(
    frame_results: list,
    atlas: "MicrostateAtlas",
) -> list | None:
    """Fast path for materialized complete 2D auto microstate grids."""

    grid_spec = _auto_complete_2d_grid_spec(atlas)
    if grid_spec is None:
        return None
    (x_name, y_name), x_edges, y_edges, by_index = grid_spec

    repartitioned: list = []
    unassigned_count = 0
    for result in frame_results:
        metadata = getattr(result, "metadata", {}) or {}
        measurements = _stored_assignment_measurements(result)
        if measurements is None or x_name not in measurements or y_name not in measurements:
            repartitioned.append(result)
            continue

        x_index = _edge_bin_index(x_edges, float(measurements[x_name]))
        y_index = _edge_bin_index(y_edges, float(measurements[y_name]))
        if x_index is None or y_index is None:
            unassigned_count += 1
            repartitioned.append(result)
            continue
        definition = by_index[(x_index, y_index)]

        updated_metadata = dict(metadata)
        updated_metadata["assignment_measurements"] = measurements
        updated_metadata["cv_centers"] = definition.cv_centers
        if definition.phi_center is not None:
            updated_metadata["phi_center"] = definition.phi_center
        if definition.psi_center is not None:
            updated_metadata["psi_center"] = definition.psi_center
        updated_metadata["plot_x"] = definition.plot_x
        updated_metadata["plot_y"] = definition.plot_y
        repartitioned.append(
            replace(
                result,
                microstate_id=definition.state_id,
                microstate_label=definition.label,
                metadata=updated_metadata,
            )
        )

    if unassigned_count > 0:
        print(
            f"[FELP] fast 2D-grid repartition: {unassigned_count} frame(s) could not "
            "be assigned to the new grid",
            file=sys.stderr,
        )
    print(
        f"[FELP] fast 2D-grid repartition: assigned {len(repartitioned)} frame(s) "
        f"using direct {len(x_edges) - 1}x{len(y_edges) - 1} binning",
        file=sys.stderr,
    )
    return repartitioned


def _repartition_frame_results_to_phi_psi_grid(
    frame_results: list,
    atlas: "MicrostateAtlas",
    *,
    phi_bins: int,
    psi_bins: int,
) -> list | None:
    """Fast path for uniform phi/psi grid repartitioning.

    The generic atlas assignment scans every definition for every frame. That is
    fine for small/custom state sets, but a 180x180 grid has 32,400 states. For
    the built-in grid we can compute the bin index directly in O(1) per frame.
    """

    if phi_bins < 1 or psi_bins < 1:
        return None
    by_id = atlas.by_id()
    if len(by_id) != phi_bins * psi_bins:
        return None
    phi_width = 360.0 / float(phi_bins)
    psi_width = 360.0 / float(psi_bins)

    repartitioned: list = []
    unassigned_count = 0
    for result in frame_results:
        metadata = getattr(result, "metadata", {}) or {}
        measurements = _stored_assignment_measurements(result)
        if measurements is None or "phi" not in measurements or "psi" not in measurements:
            repartitioned.append(result)
            continue

        try:
            phi_value = float(measurements["phi"])
            psi_value = float(measurements["psi"])
            phi_bin = int(((phi_value + 180.0) % 360.0) / phi_width)
            psi_bin = int(((psi_value + 180.0) % 360.0) / psi_width)
            phi_bin = min(max(phi_bin, 0), phi_bins - 1)
            psi_bin = min(max(psi_bin, 0), psi_bins - 1)
            definition = by_id[f"{phi_bin}_{psi_bin}"]
        except Exception:
            unassigned_count += 1
            repartitioned.append(result)
            continue

        updated_metadata = dict(metadata)
        updated_metadata["assignment_measurements"] = measurements
        updated_metadata["cv_centers"] = definition.cv_centers
        if definition.phi_center is not None:
            updated_metadata["phi_center"] = definition.phi_center
        if definition.psi_center is not None:
            updated_metadata["psi_center"] = definition.psi_center
        updated_metadata["plot_x"] = definition.plot_x
        updated_metadata["plot_y"] = definition.plot_y
        repartitioned.append(
            replace(
                result,
                microstate_id=definition.state_id,
                microstate_label=definition.label,
                metadata=updated_metadata,
            )
        )

    if unassigned_count > 0:
        print(
            f"[FELP] fast grid repartition: {unassigned_count} frame(s) could not "
            "be assigned to the new grid",
            file=sys.stderr,
        )
    print(
        f"[FELP] fast grid repartition: assigned {len(repartitioned)} frame(s) "
        f"using direct {phi_bins}x{psi_bins} binning",
        file=sys.stderr,
    )
    return repartitioned


def _build_auto_atlas_from_frame_results(
    frame_results: list,
    microstate_config,
    output_dir: str,
) -> "MicrostateAtlas":
    """Build an auto-generated atlas directly from stored CV measurements."""

    from felp.reweighting import build_auto_microstate_payload

    import numpy as np

    cv_names: tuple[str, str] | None = None
    cv_arrays: dict[str, list[float]] = {}
    for result in frame_results:
        measurements = _stored_assignment_measurements(result)
        if measurements is None:
            continue
        if cv_names is None:
            names = tuple(measurements)
            if len(names) != 2:
                raise ConfigurationError(
                    "Repartition with auto microstates requires exactly 2 CVs in the "
                    f"stored frame results, but found {len(names)}: "
                    f"{', '.join(names) or '<none>'}."
                )
            cv_names = names  # type: ignore[assignment]
            cv_arrays = {name: [] for name in cv_names}
        if not all(name in measurements for name in cv_names):
            continue
        for name in cv_names:
            cv_arrays.setdefault(name, []).append(float(measurements[name]))

    if cv_names is None:
        raise ConfigurationError(
            "Repartition with auto microstates requires stored assignment_measurements, "
            "but no usable frame measurements were found."
        )

    cv_values = {
        name: np.asarray(values, dtype=float) for name, values in cv_arrays.items()
    }
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

    file_config = replace(
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
        grid_bins = (
            (microstate_config.phi_bins, microstate_config.psi_bins)
            if microstate_config.source == "grid"
            else None
        )
        frame_results = _repartition_frame_results(
            frame_results,
            atlas,
            grid_bins=grid_bins,
        )
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
        if not repartition
        and sampling_manifest is not None
        and sampling_manifest.skipped_frame_count > 0
        else None
    )
    sampling_metadata = (
        sampling_manifest.metadata() if sampling_manifest is not None else None
    )

    reweighting_config = build_reweighting_config(namespace)
    orchestrator = ReweightingOrchestrator(
        temperature_kelvin=namespace.temperature_kelvin,
        config=reweighting_config,
        qc_config=build_reweighting_qc_config(namespace),
        workers=getattr(namespace, "workers", 1),
    )
    methods = reweighting_config.methods
    if len(methods) == 1:
        reweighting_outcome = orchestrator.run_single(
            frame_results,
            atlas,
            reference_counts=reference_counts,
            sampling_metadata=sampling_metadata,
        )
        microstate_results = reweighting_outcome.microstate_results
    else:
        outcomes = orchestrator.run(
            frame_results,
            atlas,
            reference_counts=reference_counts,
            sampling_metadata=sampling_metadata,
            output_dir=namespace.output_dir,
            writer=writer,
        )
        reweighting_outcome = outcomes[methods[0]]
        microstate_results = reweighting_outcome.microstate_results
        writer.write_microstate_results(microstate_results)
    if repartition:
        writer.write_frame_results(frame_results)
    if len(methods) == 1:
        writer.write_microstate_results(microstate_results)
        frame_weights = reweighting_outcome.auxiliary.get("frame_weights")
        if frame_weights:
            writer.write_frame_weights(frame_weights)
    summary = _build_reweighting_summary(
        frame_count=len(frame_results),
        reweighted_frame_count=sum(
            1 for result in frame_results if is_reweighting_eligible_frame(result)
        ),
        microstate_results=microstate_results,
        output_dir=namespace.output_dir,
    )
    summary.update(
        {
            "reweighting_method": reweighting_outcome.method_id,
            "reweighting_methods": list(methods),
            "state_reweighting_estimator": reweighting_outcome.state_reweighting_estimator,
        }
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


def plot_md_kl(namespace) -> dict:
    """Generate standalone KL plots comparing MD to reweighted FELP data."""

    summary = generate_md_kl_plot(build_md_kl_plot_config(namespace))
    _write_stdout(summary)
    return summary


def compare_ess_outputs(namespace) -> dict:
    """Generate ESS and overlap comparison plots across FELP runs."""

    summary = generate_ess_comparison_plots(build_ess_compare_config(namespace))
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
    if namespace.command == "md-kl-plot":
        plot_md_kl(namespace)
        return 0
    if namespace.command == "ess-compare":
        compare_ess_outputs(namespace)
        return 0
    parser.error(f"Unknown command: {namespace.command}")
    return 2
