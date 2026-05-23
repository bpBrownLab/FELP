"""Shared finalization from per-state factors to MicrostateResult rows."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from felp.core import (
    K_BOLTZMANN_KJ_PER_MOL_K,
    MicrostateResult,
    ReweightingQCConfig,
    ReweightingError,
)
from felp.reweighting.aggregator import (
    _LOW_ESS_THRESHOLDS,
    _effective_sample_size,
    _state_qc_thresholds,
    _top_weight_mass,
)
from felp.reweighting.overlap_support import (
    LOW_ESS_RELATIVE_FRACTION,
    state_has_unreliable_local_support,
)
from felp.reweighting.outcome import ReweightingOutcome
from felp.reweighting.prep import ReweightingPrep, StatePrepData


def _qc_state_summary(
    item: dict[str, object],
    target_probability: float,
) -> dict[str, object]:
    report = item["final_qc_report"]
    return {
        "state_id": item["definition"].state_id,
        "label": item["definition"].label,
        "target_probability": float(target_probability),
        "reference_probability": float(item["reference_probability"]),
        "state_qc_status": report.status,
        "dominant_frame_index": report.dominant_frame_index,
        "dominant_frame_weight_fraction": report.dominant_frame_weight_fraction,
        "state_effective_sample_size": report.effective_sample_size,
        "pareto_k": report.pareto_k,
        "pareto_k_threshold": report.pareto_k_threshold,
        "excluded_frame_indices": list(item["excluded_frame_indices"]),
    }


def finalize_microstate_outcome(
    prep: ReweightingPrep,
    *,
    method_id: str,
    state_reweighting_estimator: str,
    reweight_factors: dict[str, float],
    per_state_metadata: dict[str, dict[str, object]] | None = None,
    sampling_metadata: dict[str, object] | None = None,
    method_metadata: dict[str, object] | None = None,
    overlap_definition: str = "discrete_microstate",
) -> ReweightingOutcome:
    """Normalize factors, compute overlap/ESS diagnostics, and build results."""

    per_state_metadata = per_state_metadata or {}
    preliminary_results: list[dict[str, object]] = []
    treated_global_weights: list[np.ndarray] = []

    for definition in prep.atlas.definitions:
        state_id = definition.state_id
        state = prep.state_data[state_id]
        reweight_factor = float(reweight_factors.get(state_id, 0.0))
        if state.frames:
            treated_global_weights.append(state.treated_weights)
        preliminary_results.append(
            {
                "definition": definition,
                "frame_count": len(state.frames),
                "original_frame_count": state.original_frame_count,
                "assigned_frame_count": state.assigned_frame_count,
                "reference_probability": state.reference_probability,
                "reweight_factor": reweight_factor,
                "effective_sample_size": state.local_ess,
                "mean_delta_u_kj_mol": state.mean_delta_u_kj_mol,
                "std_delta_u_kj_mol": state.std_delta_u_kj_mol,
                "mean_boltzmann_factor": state.mean_boltzmann_factor,
                "std_boltzmann_factor": state.std_boltzmann_factor,
                "original_qc_report": state.original_qc_report,
                "final_qc_report": state.final_qc_report,
                "excluded_frame_indices": state.excluded_frame_indices,
                "psis_weight_metadata": state.psis_weight_metadata,
                "method_metadata": dict(per_state_metadata.get(state_id, {})),
            }
        )

    global_weights = (
        np.concatenate(treated_global_weights)
        if treated_global_weights
        else np.asarray([], dtype=float)
    )
    global_weight_sum = float(global_weights.sum())
    normalized_global_weights = (
        global_weights / global_weight_sum
        if global_weight_sum > 0.0
        else np.zeros_like(global_weights)
    )
    global_effective_sample_size = _effective_sample_size(global_weights)

    normalization = sum(
        item["reference_probability"] * item["reweight_factor"]
        for item in preliminary_results
    )
    if normalization <= 0.0:
        raise ReweightingError(
            "The reweighted normalization constant is non-positive. "
            "Check the chosen prior mode and frame support."
        )

    target_probabilities = np.asarray(
        [
            item["reference_probability"] * item["reweight_factor"] / normalization
            for item in preliminary_results
        ],
        dtype=float,
    )
    reference_probability_vector = np.asarray(
        [item["reference_probability"] for item in preliminary_results],
        dtype=float,
    )
    overlap_coefficient = float(
        np.minimum(reference_probability_vector, target_probabilities).sum()
    )
    target_probability_low_ess_mass = {
        threshold: float(
            sum(
                target_probability
                for item, target_probability in zip(
                    preliminary_results,
                    target_probabilities,
                    strict=True,
                )
                if item["effective_sample_size"] < threshold
            )
        )
        for threshold in _LOW_ESS_THRESHOLDS
    }
    target_probability_unreliable_local_support = float(
        sum(
            target_probability
            for item, target_probability in zip(
                preliminary_results,
                target_probabilities,
                strict=True,
            )
            if state_has_unreliable_local_support(
                float(item["effective_sample_size"]),
                int(item["frame_count"]),
                min_state_ess=prep.qc_config.min_state_ess,
            )
        )
    )
    qc_top_items = [
        (item, target_probability)
        for item, target_probability in zip(
            preliminary_results, target_probabilities, strict=True
        )
        if item["original_qc_report"].status == "flagged"
        or item["final_qc_report"].status == "filtered"
    ]
    qc_config = prep.qc_config
    reweighting_qc_summary = {
        "policy": qc_config.policy,
        "method": qc_config.method,
        "weight_treatment": qc_config.weight_treatment,
        "thresholds": _state_qc_thresholds(qc_config),
        "flagged_state_count": sum(
            1
            for item in preliminary_results
            if item["original_qc_report"].status == "flagged"
        ),
        "filtered_state_count": sum(
            1 for item in preliminary_results if item["excluded_frame_indices"]
        ),
        "excluded_frame_count": sum(
            len(item["excluded_frame_indices"]) for item in preliminary_results
        ),
        "psis_smoothed_state_count": sum(
            1
            for item in preliminary_results
            if item["psis_weight_metadata"].get("psis_smoothed")
        ),
        "psis_smoothed_frame_count": sum(
            int(item["psis_weight_metadata"].get("psis_smoothed_frame_count", 0))
            for item in preliminary_results
        ),
        "target_probability_flagged_state_mass": float(
            sum(target_probability for _, target_probability in qc_top_items)
        ),
        "top_flagged_states_by_target_probability": [
            _qc_state_summary(item, target_probability)
            for item, target_probability in sorted(
                qc_top_items,
                key=lambda value: value[1],
                reverse=True,
            )[:10]
        ],
        "top_flagged_states_by_dominant_frame_fraction": [
            _qc_state_summary(item, target_probability)
            for item, target_probability in sorted(
                qc_top_items,
                key=lambda value: (
                    value[0]["final_qc_report"].dominant_frame_weight_fraction
                ),
                reverse=True,
            )[:10]
        ],
    }
    shared_metadata: dict[str, object] = {
        "reweighting_method": method_id,
        "state_reweighting_estimator": state_reweighting_estimator,
        "overlap_definition": overlap_definition,
        "delta_shift_kj_mol": prep.global_delta_shift,
        "reference_probability_source": prep.reference_probability_source,
        "prior_probability_mode": prep.atlas.reference_probability_mode,
        "unsupported_state_count": 0,
        "unsupported_state_ids": [],
        "eligible_frame_count": prep.eligible_frame_count,
        "filtered_eligible_frame_count": prep.filtered_eligible_frame_count,
        "reference_count_scope": (
            "full_assigned_frames"
            if prep.reference_counts is not None
            else "evaluated_eligible_frames"
        ),
        "global_effective_sample_size": global_effective_sample_size,
        "global_effective_sample_fraction": (
            global_effective_sample_size / prep.filtered_eligible_frame_count
        ),
        "max_normalized_weight": (
            float(normalized_global_weights.max())
            if normalized_global_weights.size
            else 0.0
        ),
        "top_1_percent_weight_mass": _top_weight_mass(normalized_global_weights, 0.01),
        "top_5_percent_weight_mass": _top_weight_mass(normalized_global_weights, 0.05),
        "top_10_percent_weight_mass": _top_weight_mass(
            normalized_global_weights, 0.10
        ),
        "overlap_coefficient": overlap_coefficient,
        "target_probability_low_ess_mass_ess_lt_10": target_probability_low_ess_mass[
            10
        ],
        "target_probability_low_ess_mass_ess_lt_50": target_probability_low_ess_mass[
            50
        ],
        "target_probability_unreliable_local_support": (
            target_probability_unreliable_local_support
        ),
        "unreliable_local_support_criterion": (
            "ESS_i < max(min_state_ess, "
            f"{LOW_ESS_RELATIVE_FRACTION:g} * n_i); "
            f"min_state_ess={prep.qc_config.min_state_ess:g}"
        ),
        "reweighting_qc_policy": qc_config.policy,
        "reweighting_weight_treatment": qc_config.weight_treatment,
        "reweighting_qc_thresholds": _state_qc_thresholds(qc_config),
        "reweighting_qc": reweighting_qc_summary,
    }
    if sampling_metadata is not None:
        shared_metadata["sampling"] = dict(sampling_metadata)
    if method_metadata:
        shared_metadata.update(method_metadata)

    microstate_results: list[MicrostateResult] = []
    for item, target_probability in zip(
        preliminary_results,
        target_probabilities,
        strict=True,
    ):
        definition = item["definition"]
        free_energy = None
        if target_probability > 0.0:
            free_energy = -(
                K_BOLTZMANN_KJ_PER_MOL_K
                * prep.temperature_kelvin
                * math.log(target_probability)
            )
        cv_centers = definition.cv_centers
        method_state_metadata = dict(item["method_metadata"])
        microstate_results.append(
            MicrostateResult(
                state_id=definition.state_id,
                label=definition.label,
                phi_center=cv_centers.get("phi"),
                psi_center=cv_centers.get("psi"),
                frame_count=item["frame_count"],
                reference_probability=item["reference_probability"],
                reweight_factor=item["reweight_factor"],
                target_probability=float(target_probability),
                free_energy_kj_mol=free_energy,
                effective_sample_size=item["effective_sample_size"],
                mean_delta_u_kj_mol=item["mean_delta_u_kj_mol"],
                std_delta_u_kj_mol=item["std_delta_u_kj_mol"],
                mean_boltzmann_factor=item["mean_boltzmann_factor"],
                std_boltzmann_factor=item["std_boltzmann_factor"],
                plot_x=definition.plot_x,
                plot_y=definition.plot_y,
                state_level=definition.state_level,
                metadata={
                    **shared_metadata,
                    "cv_ranges": dict(definition.cv_ranges),
                    "cv_centers": cv_centers,
                    "assigned_frame_count": item["assigned_frame_count"],
                    "evaluated_frame_count": item["original_frame_count"],
                    "filtered_evaluated_frame_count": item["frame_count"],
                    "excluded_frame_indices": list(item["excluded_frame_indices"]),
                    **item["final_qc_report"].to_metadata(),
                    "original_state_qc": item["original_qc_report"].to_metadata(),
                    "weight_treatment": qc_config.weight_treatment,
                    "psis_weight_smoothing": item["psis_weight_metadata"],
                    **method_state_metadata,
                },
            )
        )

    finite_energies = [
        result.free_energy_kj_mol
        for result in microstate_results
        if result.free_energy_kj_mol is not None
    ]
    if finite_energies:
        min_energy = min(finite_energies)
        microstate_results = [
            MicrostateResult(
                state_id=result.state_id,
                label=result.label,
                phi_center=result.phi_center,
                psi_center=result.psi_center,
                frame_count=result.frame_count,
                reference_probability=result.reference_probability,
                reweight_factor=result.reweight_factor,
                target_probability=result.target_probability,
                free_energy_kj_mol=(
                    None
                    if result.free_energy_kj_mol is None
                    else result.free_energy_kj_mol - min_energy
                ),
                effective_sample_size=result.effective_sample_size,
                mean_delta_u_kj_mol=result.mean_delta_u_kj_mol,
                std_delta_u_kj_mol=result.std_delta_u_kj_mol,
                mean_boltzmann_factor=result.mean_boltzmann_factor,
                std_boltzmann_factor=result.std_boltzmann_factor,
                plot_x=result.plot_x,
                plot_y=result.plot_y,
                state_level=result.state_level,
                metadata=result.metadata,
            )
            for result in microstate_results
        ]

    return ReweightingOutcome(
        method_id=method_id,
        microstate_results=microstate_results,
        shared_metadata=shared_metadata,
    )
