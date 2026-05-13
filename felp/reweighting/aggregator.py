"""Local reweighting aggregation for FELP."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
from typing import Any, Iterable

import numpy as np

from felp.core import (
    angular_midpoint_degrees,
    FrameResult,
    K_BOLTZMANN_KJ_PER_MOL_K,
    MicrostateResult,
    ReweightingQCConfig,
    ReweightingError,
)
from .atlas import MicrostateAtlas

_LOW_ESS_THRESHOLDS = (10, 50)
_ANGULAR_CV_NAMES = {"phi", "psi", "omega"}
_PSIS_MIN_TAIL_SAMPLES = 3
_PSIS_CONSERVATIVE_K_THRESHOLD = 0.7


def is_reweighting_eligible_frame(result: FrameResult) -> bool:
    """Return whether a frame can contribute to the reweighting estimate."""

    if result.metadata.get("status") == "failed":
        return False
    if not (
        math.isfinite(result.reference_energy_kj_mol)
        and math.isfinite(result.perturbed_energy_kj_mol)
        and math.isfinite(result.delta_u_kj_mol)
    ):
        return False
    return True


def _reference_probability_map(
    atlas: MicrostateAtlas,
    grouped: dict[str, list[FrameResult]],
    empirical_total: int,
    reference_counts: dict[str, int] | None = None,
) -> tuple[dict[str, float], str]:
    """Return the normalized reference prior for each state."""

    mode = atlas.reference_probability_mode
    if mode == "loaded":
        missing = [
            definition.state_id
            for definition in atlas.definitions
            if definition.reference_probability is None
        ]
        if missing:
            missing_text = ", ".join(missing[:10])
            if len(missing) > 10:
                missing_text = f"{missing_text}, ..."
            raise ReweightingError(
                "Loaded-prior reweighting requires reference_probability for every "
                f"state; missing: {missing_text}."
            )
        negative = [
            definition.state_id
            for definition in atlas.definitions
            if float(definition.reference_probability or 0.0) < 0.0
        ]
        if negative:
            raise ReweightingError(
                "Reference probabilities must be non-negative; invalid states: "
                + ", ".join(negative[:10])
            )
        normalization = sum(
            float(definition.reference_probability or 0.0)
            for definition in atlas.definitions
        )
        if normalization <= 0.0:
            raise ReweightingError(
                "Loaded-prior reweighting requires a positive total reference "
                "probability."
            )
        reference_probabilities = {
            definition.state_id: float(definition.reference_probability or 0.0)
            / normalization
            for definition in atlas.definitions
        }
        return reference_probabilities, "loaded"

    if mode == "raw_counts":
        count_source = reference_counts if reference_counts is not None else None
        if count_source is not None:
            empirical_total = sum(
                int(count_source.get(definition.state_id, 0))
                for definition in atlas.definitions
            )
        if empirical_total <= 0:
            raise ReweightingError(
                "Raw-count prior reweighting requires at least one eligible frame."
            )
        reference_probabilities = {
            definition.state_id: (
                (
                    int(count_source.get(definition.state_id, 0))
                    if count_source is not None
                    else len(grouped.get(definition.state_id, []))
                )
                / empirical_total
            )
            for definition in atlas.definitions
        }
        return reference_probabilities, "raw_counts"

    raise ReweightingError(
        "Unsupported reference_probability_mode: "
        f"{atlas.reference_probability_mode}"
    )


def _unsupported_positive_prior_states(
    atlas: MicrostateAtlas,
    reference_probabilities: dict[str, float],
    grouped: dict[str, list[FrameResult]],
) -> list[str]:
    """Return nonzero-prior states that have no evaluated frame support."""

    return [
        definition.state_id
        for definition in atlas.definitions
        if reference_probabilities.get(definition.state_id, 0.0) > 0.0
        and not grouped.get(definition.state_id)
    ]


def _effective_sample_size(weights: np.ndarray) -> float:
    """Return the standard importance-sampling ESS for a weight vector."""

    if weights.size == 0:
        return 0.0
    return float((weights.sum() ** 2) / max(np.square(weights).sum(), 1.0e-12))


def _top_weight_mass(normalized_weights: np.ndarray, fraction: float) -> float:
    """Return the cumulative mass of the largest-weight fraction of frames."""

    if normalized_weights.size == 0:
        return 0.0
    count = max(1, math.ceil(normalized_weights.size * fraction))
    ordered = np.sort(normalized_weights)[::-1]
    return float(ordered[:count].sum())


def _state_local_weights(
    frames: list[FrameResult],
    beta: float,
) -> tuple[np.ndarray, float | None]:
    """Return numerically stable in-state Boltzmann weights and local shift."""

    if not frames:
        return np.asarray([], dtype=float), None
    delta_values = np.asarray([frame.delta_u_kj_mol for frame in frames], dtype=float)
    delta_shift = float(delta_values.min())
    return np.exp(-beta * (delta_values - delta_shift)), delta_shift


def _state_delta_g_kj_mol(frames: list[FrameResult], beta: float) -> float | None:
    """Return the unnormalized state Zwanzig free-energy estimate."""

    weights, delta_shift = _state_local_weights(frames, beta)
    if weights.size == 0 or delta_shift is None:
        return None
    mean_weight = float(weights.mean())
    if mean_weight <= 0.0:
        return None
    return float(delta_shift - math.log(mean_weight) / beta)


def _psis_tail_count(sample_count: int) -> int:
    """Return the PSIS-style upper-tail size used for Pareto diagnostics."""

    if sample_count <= _PSIS_MIN_TAIL_SAMPLES:
        return 0
    return min(
        sample_count - 1,
        max(
            _PSIS_MIN_TAIL_SAMPLES,
            min(sample_count // 5, int(3.0 * math.sqrt(sample_count))),
        ),
    )


def _psis_k_threshold(sample_count: int) -> float | None:
    """Return the sample-size dependent Pareto-k reliability threshold."""

    if sample_count <= 10:
        return None
    return min(
        _PSIS_CONSERVATIVE_K_THRESHOLD,
        1.0 - 1.0 / math.log10(float(sample_count)),
    )


def _pareto_tail_shape(weights: np.ndarray) -> tuple[float | None, int]:
    """Estimate the upper-tail Pareto shape using a Hill tail estimator."""

    finite_weights = np.asarray(weights[np.isfinite(weights)], dtype=float)
    positive_weights = finite_weights[finite_weights > 0.0]
    tail_count = _psis_tail_count(int(positive_weights.size))
    if tail_count == 0:
        return None, 0

    ordered = np.sort(positive_weights)
    tail = ordered[-tail_count:]
    threshold = ordered[-tail_count - 1]
    if threshold <= 0.0:
        return float("inf"), tail_count
    log_ratios = np.log(tail / threshold)
    if not np.isfinite(log_ratios).all():
        return float("inf"), tail_count
    return max(0.0, float(log_ratios.mean())), tail_count


def psis_smooth_weights(weights: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    """Return Pareto-smoothed importance weights plus smoothing metadata.

    The input weights can be in any common scale. Only the upper tail is modified,
    and a finite-sample truncation cap is applied after smoothing to control
    residual variance from single-frame spikes.
    """

    raw_weights = np.asarray(weights, dtype=float)
    smoothed = raw_weights.copy()
    finite_positive = np.isfinite(raw_weights) & (raw_weights > 0.0)
    positive_count = int(finite_positive.sum())
    pareto_k, tail_count = _pareto_tail_shape(raw_weights)
    metadata: dict[str, object] = {
        "psis_smoothed": False,
        "psis_smoothed_frame_count": 0,
        "psis_pareto_k": pareto_k,
        "psis_tail_count": tail_count,
        "psis_truncation_cap": None,
    }
    if tail_count == 0 or positive_count <= tail_count:
        return smoothed, metadata

    positive_indices = np.flatnonzero(finite_positive)
    positive_weights = raw_weights[positive_indices]
    order = np.argsort(positive_weights)
    ordered_weights = positive_weights[order]
    threshold = float(ordered_weights[-tail_count - 1])
    if threshold <= 0.0 or pareto_k is None or not math.isfinite(pareto_k):
        return smoothed, metadata

    tail_local_indices = order[-tail_count:]
    tail_weights = positive_weights[tail_local_indices]
    tail_order = np.argsort(tail_weights)
    tail_weights_ordered = tail_weights[tail_order]

    probabilities = (np.arange(1, tail_count + 1, dtype=float) - 0.5) / tail_count
    smoothed_tail = threshold * np.power(
        np.maximum(1.0 - probabilities, 1.0e-12),
        -max(float(pareto_k), 1.0e-12),
    )
    # Do not let the fitted tail increase any observed order statistic.
    smoothed_tail = np.minimum(smoothed_tail, tail_weights_ordered)

    tail_indices_ordered = positive_indices[tail_local_indices[tail_order]]
    smoothed[tail_indices_ordered] = smoothed_tail

    mean_weight = float(smoothed[finite_positive].mean())
    truncation_cap = (positive_count**0.75) * mean_weight
    if math.isfinite(truncation_cap) and truncation_cap > 0.0:
        smoothed[finite_positive] = np.minimum(smoothed[finite_positive], truncation_cap)
        metadata["psis_truncation_cap"] = float(truncation_cap)

    metadata["psis_smoothed"] = True
    metadata["psis_smoothed_frame_count"] = tail_count
    return smoothed, metadata


@dataclass(frozen=True)
class _StateQCReport:
    status: str
    reasons: tuple[str, ...]
    frame_count: int
    effective_sample_size: float
    effective_sample_fraction: float
    max_normalized_weight: float
    dominant_frame_index: int | None
    dominant_frame_delta_u_kj_mol: float | None
    dominant_frame_weight_fraction: float
    leave_one_out_delta_g_shift_kj_mol: float | None
    pareto_k: float | None
    pareto_k_threshold: float | None
    pareto_tail_count: int

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-serializable metadata for CSV summaries."""

        return {
            "state_qc_status": self.status,
            "state_qc_reasons": list(self.reasons),
            "state_effective_sample_size": self.effective_sample_size,
            "state_effective_sample_fraction": self.effective_sample_fraction,
            "state_max_normalized_weight": self.max_normalized_weight,
            "dominant_frame_index": self.dominant_frame_index,
            "dominant_frame_delta_u_kj_mol": self.dominant_frame_delta_u_kj_mol,
            "dominant_frame_weight_fraction": self.dominant_frame_weight_fraction,
            "leave_one_out_delta_g_shift_kj_mol": (
                self.leave_one_out_delta_g_shift_kj_mol
            ),
            "pareto_k": self.pareto_k,
            "pareto_k_threshold": self.pareto_k_threshold,
            "pareto_tail_count": self.pareto_tail_count,
        }


def _state_qc_thresholds(config: ReweightingQCConfig) -> dict[str, object]:
    """Return a stable threshold summary for result metadata."""

    return {
        "min_frame_count": config.min_frame_count,
        "method": config.method,
        "weight_treatment": config.weight_treatment,
        "max_frame_weight_fraction": config.max_frame_weight_fraction,
        "min_state_ess": config.min_state_ess,
        "max_leave_one_out_shift_kj_mol": (
            config.max_leave_one_out_shift_kj_mol
        ),
        "max_excluded_frames_per_state": config.max_excluded_frames_per_state,
    }


def _state_qc_report(
    frames: list[FrameResult],
    *,
    beta: float,
    config: ReweightingQCConfig,
) -> _StateQCReport:
    """Diagnose whether one frame dominates a state's Zwanzig estimator."""

    weights, _ = _state_local_weights(frames, beta)
    frame_count = len(frames)
    if weights.size == 0:
        return _StateQCReport(
            status="ok",
            reasons=(),
            frame_count=0,
            effective_sample_size=0.0,
            effective_sample_fraction=0.0,
            max_normalized_weight=0.0,
            dominant_frame_index=None,
            dominant_frame_delta_u_kj_mol=None,
            dominant_frame_weight_fraction=0.0,
            leave_one_out_delta_g_shift_kj_mol=None,
            pareto_k=None,
            pareto_k_threshold=None,
            pareto_tail_count=0,
        )

    weight_sum = float(weights.sum())
    normalized_weights = (
        weights / weight_sum if weight_sum > 0.0 else np.zeros_like(weights)
    )
    dominant_index = int(np.argmax(normalized_weights))
    dominant_frame = frames[dominant_index]
    ess = _effective_sample_size(weights)
    max_weight = float(normalized_weights[dominant_index])
    full_delta_g = _state_delta_g_kj_mol(frames, beta)
    leave_one_out_shift = None
    if frame_count > 1 and full_delta_g is not None:
        remaining = [
            frame for index, frame in enumerate(frames) if index != dominant_index
        ]
        leave_one_out_delta_g = _state_delta_g_kj_mol(remaining, beta)
        if leave_one_out_delta_g is not None:
            leave_one_out_shift = abs(leave_one_out_delta_g - full_delta_g)

    pareto_k, pareto_tail_count = _pareto_tail_shape(weights)
    pareto_k_threshold = _psis_k_threshold(frame_count)
    reasons: list[str] = []
    if (
        config.policy != "none"
        and config.method == "auto"
        and pareto_k is not None
        and pareto_k_threshold is not None
        and pareto_k > pareto_k_threshold
    ):
        reasons.append(
            "Pareto tail shape exceeds automatic PSIS reliability threshold "
            f"({pareto_k:.3g} > {pareto_k_threshold:.3g})"
        )
    elif (
        config.policy != "none"
        and config.method == "threshold"
        and frame_count >= config.min_frame_count
    ):
        if max_weight > config.max_frame_weight_fraction:
            reasons.append(
                "dominant frame weight fraction exceeds "
                f"{config.max_frame_weight_fraction:g}"
            )
        if ess < config.min_state_ess:
            reasons.append(f"state ESS is below {config.min_state_ess:g}")
        if (
            leave_one_out_shift is not None
            and leave_one_out_shift > config.max_leave_one_out_shift_kj_mol
        ):
            reasons.append(
                "leave-one-out free-energy shift exceeds "
                f"{config.max_leave_one_out_shift_kj_mol:g} kJ/mol"
            )

    return _StateQCReport(
        status="flagged" if reasons else "ok",
        reasons=tuple(reasons),
        frame_count=frame_count,
        effective_sample_size=ess,
        effective_sample_fraction=ess / frame_count if frame_count else 0.0,
        max_normalized_weight=max_weight,
        dominant_frame_index=dominant_frame.frame_index,
        dominant_frame_delta_u_kj_mol=float(dominant_frame.delta_u_kj_mol),
        dominant_frame_weight_fraction=max_weight,
        leave_one_out_delta_g_shift_kj_mol=leave_one_out_shift,
        pareto_k=pareto_k,
        pareto_k_threshold=pareto_k_threshold,
        pareto_tail_count=pareto_tail_count,
    )


def _apply_state_qc_policy(
    frames: list[FrameResult],
    *,
    beta: float,
    config: ReweightingQCConfig,
) -> tuple[list[FrameResult], _StateQCReport, _StateQCReport, tuple[int, ...]]:
    """Return possibly filtered frames plus original/final QC reports."""

    original_report = _state_qc_report(frames, beta=beta, config=config)
    if config.policy != "exclude-frames" or original_report.status != "flagged":
        return frames, original_report, original_report, ()

    remaining = list(frames)
    excluded: list[int] = []
    final_report = original_report
    while (
        final_report.status == "flagged"
        and final_report.dominant_frame_index is not None
        and len(excluded) < config.max_excluded_frames_per_state
    ):
        dominant_frame_index = final_report.dominant_frame_index
        excluded.append(dominant_frame_index)
        remaining = [
            frame for frame in remaining if frame.frame_index != dominant_frame_index
        ]
        final_report = _state_qc_report(remaining, beta=beta, config=config)

    if excluded:
        final_report = _StateQCReport(
            status="filtered",
            reasons=final_report.reasons,
            frame_count=final_report.frame_count,
            effective_sample_size=final_report.effective_sample_size,
            effective_sample_fraction=final_report.effective_sample_fraction,
            max_normalized_weight=final_report.max_normalized_weight,
            dominant_frame_index=final_report.dominant_frame_index,
            dominant_frame_delta_u_kj_mol=final_report.dominant_frame_delta_u_kj_mol,
            dominant_frame_weight_fraction=final_report.dominant_frame_weight_fraction,
            leave_one_out_delta_g_shift_kj_mol=(
                final_report.leave_one_out_delta_g_shift_kj_mol
            ),
            pareto_k=final_report.pareto_k,
            pareto_k_threshold=final_report.pareto_k_threshold,
            pareto_tail_count=final_report.pareto_tail_count,
        )
    return remaining, original_report, final_report, tuple(excluded)


def _cv_range_centers(cv_ranges: dict[str, tuple[float, float]]) -> dict[str, float]:
    """Return display centers for arbitrary CV ranges."""

    centers: dict[str, float] = {}
    for name, bounds in cv_ranges.items():
        lower, upper = (float(value) for value in bounds)
        if name in _ANGULAR_CV_NAMES:
            centers[name] = angular_midpoint_degrees(lower, upper)
        else:
            centers[name] = (lower + upper) / 2.0
    return centers


@dataclass
class ReweightingAggregator:
    """Aggregate per-frame deltaU values into reweighted microstate populations."""

    temperature_kelvin: float
    qc_config: ReweightingQCConfig = field(default_factory=ReweightingQCConfig)

    @property
    def beta(self) -> float:
        return 1.0 / (K_BOLTZMANN_KJ_PER_MOL_K * self.temperature_kelvin)

    def aggregate(
        self,
        frame_results: Iterable[FrameResult],
        atlas: MicrostateAtlas,
        *,
        reference_counts: dict[str, int] | None = None,
        sampling_metadata: dict[str, object] | None = None,
    ) -> list[MicrostateResult]:
        """Compute Zwanzig reweighting factors and reconstruct target state populations."""

        frame_results = list(frame_results)
        if not frame_results:
            return []

        eligible_frames = [
            result for result in frame_results if is_reweighting_eligible_frame(result)
        ]
        if not eligible_frames:
            return []

        grouped: dict[str, list[FrameResult]] = defaultdict(list)
        for result in eligible_frames:
            grouped[result.microstate_id].append(result)

        reference_probabilities, reference_probability_source = _reference_probability_map(
            atlas,
            grouped,
            len(eligible_frames),
            reference_counts=reference_counts,
        )
        unsupported_state_ids = _unsupported_positive_prior_states(
            atlas,
            reference_probabilities,
            grouped,
        )
        if unsupported_state_ids:
            unsupported_text = ", ".join(unsupported_state_ids[:10])
            if len(unsupported_state_ids) > 10:
                unsupported_text = f"{unsupported_text}, ..."
            raise ReweightingError(
                "Cannot reweight states with positive reference prior but zero "
                f"evaluated frames: {unsupported_text}."
            )

        qc_by_state: dict[str, dict[str, object]] = {}
        filtered_grouped: dict[str, list[FrameResult]] = {}
        for definition in atlas.definitions:
            original_state_frames = list(grouped.get(definition.state_id, []))
            (
                filtered_state_frames,
                original_qc_report,
                final_qc_report,
                excluded_frame_indices,
            ) = _apply_state_qc_policy(
                original_state_frames,
                beta=self.beta,
                config=self.qc_config,
            )
            filtered_grouped[definition.state_id] = filtered_state_frames
            qc_by_state[definition.state_id] = {
                "original_qc_report": original_qc_report,
                "final_qc_report": final_qc_report,
                "excluded_frame_indices": excluded_frame_indices,
                "original_frame_count": len(original_state_frames),
            }

        filtered_eligible_frames = [
            frame
            for definition in atlas.definitions
            for frame in filtered_grouped.get(definition.state_id, [])
        ]
        if not filtered_eligible_frames:
            raise ReweightingError(
                "No eligible frames remain after reweighting QC filtering."
            )

        delta_shift = min(result.delta_u_kj_mol for result in filtered_eligible_frames)

        preliminary_results = []
        treated_global_weights: list[np.ndarray] = []
        for definition in atlas.definitions:
            state_frames = filtered_grouped.get(definition.state_id, [])
            qc_state = qc_by_state[definition.state_id]
            original_qc_report = qc_state["original_qc_report"]
            final_qc_report = qc_state["final_qc_report"]
            excluded_frame_indices = qc_state["excluded_frame_indices"]
            original_frame_count = int(qc_state["original_frame_count"])
            if state_frames:
                delta_values = np.asarray(
                    [frame.delta_u_kj_mol for frame in state_frames],
                    dtype=float,
                )
                local_weights, local_delta_shift = _state_local_weights(
                    state_frames,
                    self.beta,
                )
                psis_metadata: dict[str, object] = {
                    "psis_smoothed": False,
                    "psis_smoothed_frame_count": 0,
                    "psis_pareto_k": None,
                    "psis_tail_count": 0,
                    "psis_truncation_cap": None,
                }
                if self.qc_config.weight_treatment == "psis-smoothed":
                    local_weights, psis_metadata = psis_smooth_weights(local_weights)
                local_scale = (
                    math.exp(-self.beta * (local_delta_shift - delta_shift))
                    if local_delta_shift is not None
                    else 0.0
                )
                weights = local_scale * local_weights
                treated_global_weights.append(weights)
                reweight_factor = float(weights.mean())
                mean_delta_u = float(delta_values.mean())
                std_delta_u = float(delta_values.std(ddof=0))
                mean_weight = float(weights.mean())
                std_weight = float(weights.std(ddof=0))
                ess = _effective_sample_size(weights)
            else:
                reweight_factor = 0.0
                mean_delta_u = None
                std_delta_u = None
                mean_weight = None
                std_weight = None
                ess = 0.0
                psis_metadata = {
                    "psis_smoothed": False,
                    "psis_smoothed_frame_count": 0,
                    "psis_pareto_k": None,
                    "psis_tail_count": 0,
                    "psis_truncation_cap": None,
                }

            preliminary_results.append(
                {
                    "definition": definition,
                    "frame_count": len(state_frames),
                    "original_frame_count": original_frame_count,
                    "assigned_frame_count": (
                        int(reference_counts.get(definition.state_id, 0))
                        if reference_counts is not None
                        else original_frame_count
                    ),
                    "reference_probability": reference_probabilities.get(
                        definition.state_id, 0.0
                    ),
                    "reweight_factor": reweight_factor,
                    "effective_sample_size": ess,
                    "mean_delta_u_kj_mol": mean_delta_u,
                    "std_delta_u_kj_mol": std_delta_u,
                    "mean_boltzmann_factor": mean_weight,
                    "std_boltzmann_factor": std_weight,
                    "original_qc_report": original_qc_report,
                    "final_qc_report": final_qc_report,
                    "excluded_frame_indices": excluded_frame_indices,
                    "psis_weight_metadata": psis_metadata,
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
        qc_top_items = [
            (item, target_probability)
            for item, target_probability in zip(
                preliminary_results, target_probabilities, strict=True
            )
            if item["original_qc_report"].status == "flagged"
            or item["final_qc_report"].status == "filtered"
        ]

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

        reweighting_qc_summary = {
            "policy": self.qc_config.policy,
            "method": self.qc_config.method,
            "weight_treatment": self.qc_config.weight_treatment,
            "thresholds": _state_qc_thresholds(self.qc_config),
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
        shared_metadata = {
            "delta_shift_kj_mol": delta_shift,
            "reference_probability_source": reference_probability_source,
            "prior_probability_mode": atlas.reference_probability_mode,
            "unsupported_state_count": 0,
            "unsupported_state_ids": [],
            "eligible_frame_count": len(eligible_frames),
            "filtered_eligible_frame_count": len(filtered_eligible_frames),
            "reference_count_scope": (
                "full_assigned_frames"
                if reference_counts is not None
                else "evaluated_eligible_frames"
            ),
            "global_effective_sample_size": global_effective_sample_size,
            "global_effective_sample_fraction": (
                global_effective_sample_size / len(filtered_eligible_frames)
            ),
            "max_normalized_weight": (
                float(normalized_global_weights.max())
                if normalized_global_weights.size
                else 0.0
            ),
            "top_1_percent_weight_mass": _top_weight_mass(
                normalized_global_weights, 0.01
            ),
            "top_5_percent_weight_mass": _top_weight_mass(
                normalized_global_weights, 0.05
            ),
            "top_10_percent_weight_mass": _top_weight_mass(
                normalized_global_weights, 0.10
            ),
            "overlap_coefficient": overlap_coefficient,
            "target_probability_low_ess_mass_ess_lt_10": (
                target_probability_low_ess_mass[10]
            ),
            "target_probability_low_ess_mass_ess_lt_50": (
                target_probability_low_ess_mass[50]
            ),
            "reweighting_qc_policy": self.qc_config.policy,
            "reweighting_weight_treatment": self.qc_config.weight_treatment,
            "reweighting_qc_thresholds": _state_qc_thresholds(self.qc_config),
            "reweighting_qc": reweighting_qc_summary,
        }
        if sampling_metadata is not None:
            shared_metadata["sampling"] = dict(sampling_metadata)
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
                    * self.temperature_kelvin
                    * math.log(target_probability)
                )
            microstate_results.append(
                MicrostateResult(
                    state_id=definition.state_id,
                    label=definition.label,
                    phi_center=definition.phi_center,
                    psi_center=definition.psi_center,
                    frame_count=item["frame_count"],
                    reference_probability=item["reference_probability"],
                    reweight_factor=item["reweight_factor"],
                    target_probability=target_probability,
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
                        "cv_centers": _cv_range_centers(definition.cv_ranges),
                        "assigned_frame_count": item["assigned_frame_count"],
                        "evaluated_frame_count": item["original_frame_count"],
                        "filtered_evaluated_frame_count": item["frame_count"],
                        "excluded_frame_indices": list(
                            item["excluded_frame_indices"]
                        ),
                        **item["final_qc_report"].to_metadata(),
                        "original_state_qc": (
                            item["original_qc_report"].to_metadata()
                        ),
                        "weight_treatment": self.qc_config.weight_treatment,
                        "psis_weight_smoothing": item["psis_weight_metadata"],
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

        return microstate_results
