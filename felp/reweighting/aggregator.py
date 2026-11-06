"""Local reweighting aggregation for FELP."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable

import numpy as np

from felp.core import (
    FrameResult,
    K_BOLTZMANN_KJ_PER_MOL_K,
    MicrostateResult,
    ReweightingConfig,
    ReweightingQCConfig,
    ReweightingError,
    frame_reweighting_delta,
)
from .atlas import MicrostateAtlas

_LOW_ESS_THRESHOLDS = (10, 50)
_PSIS_K_CI_METHOD = "outer_resampling_required"


def is_reweighting_eligible_frame(result: FrameResult) -> bool:
    """Return whether a frame can contribute to the reweighting estimate."""

    if result.metadata.get("status") == "failed":
        return False
    if not (
        math.isfinite(result.reference_energy_kj_mol)
        and math.isfinite(result.perturbed_energy_kj_mol)
        and math.isfinite(result.delta_u_kj_mol)
        and math.isfinite(frame_reweighting_delta(result))
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

    weights = np.asarray(weights, dtype=float)
    if weights.size == 0:
        return 0.0
    scale = float(np.max(weights))
    if scale <= 0.0:
        return 0.0
    scaled = weights / scale
    return float(scaled.sum() ** 2 / np.square(scaled).sum())


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
    delta_values = np.asarray(
        [frame_reweighting_delta(frame) for frame in frames], dtype=float
    )
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


def _pareto_k_regime(shape_k: float | None) -> str | None:
    """Classify the fitted tail shape by which moments provably exist.

    ``k < 0.5`` implies a finite-variance importance ratio (CLT applies);
    ``0.5 <= k < 1`` implies infinite variance but a finite mean (slower,
    still-convergent estimates); ``k >= 1`` implies the mean itself does
    not exist and the estimator cannot converge.
    """

    if shape_k is None or not math.isfinite(float(shape_k)):
        return None
    value = float(shape_k)
    if value < 0.5:
        return "finite_variance"
    if value < 1.0:
        return "infinite_variance_finite_mean"
    return "infinite_mean"


@dataclass(frozen=True)
class _ParetoTailFit:
    """Numerically stable generalized-Pareto fit to an upper weight tail."""

    shape_k: float | None
    scale: float | None
    threshold: float | None
    tail_count: int
    status: str
    tail_indices: tuple[int, ...] = ()
    normalized_scale: float | None = None
    normalized_threshold: float | None = None
    weight_scale: float | None = None

    def to_metadata(self) -> dict[str, object]:
        """Return fit metadata without claiming an unsupported analytic k CI."""

        return {
            "shape_k": self.shape_k,
            "scale": self.scale,
            "threshold": self.threshold,
            "tail_count": self.tail_count,
            "fit_status": self.status,
            "k_regime": _pareto_k_regime(self.shape_k),
            "k_ci_lower": None,
            "k_ci_upper": None,
            "k_ci_method": _PSIS_K_CI_METHOD,
        }


def _psis_metadata(
    fit: _ParetoTailFit,
    *,
    smoothed: bool = False,
    smoothed_frame_count: int = 0,
    truncation_cap: float | None = None,
) -> dict[str, object]:
    """Return the stable public metadata shape for PSIS treatment."""

    return {
        "psis_smoothed": smoothed,
        "psis_smoothed_frame_count": smoothed_frame_count,
        "psis_pareto_k": fit.shape_k,
        "psis_pareto_scale": fit.scale,
        "psis_pareto_threshold": fit.threshold,
        "psis_tail_count": fit.tail_count,
        "psis_pareto_fit_status": fit.status,
        "psis_pareto_k_ci_lower": None,
        "psis_pareto_k_ci_upper": None,
        "psis_pareto_k_ci_method": _PSIS_K_CI_METHOD,
        "psis_truncation_cap": truncation_cap,
    }


def _psis_not_requested_metadata() -> dict[str, object]:
    """Return PSIS metadata for a state whose weights were left raw."""

    return _psis_metadata(
        _ParetoTailFit(
            shape_k=None,
            scale=None,
            threshold=None,
            tail_count=0,
            status="not_requested",
        )
    )


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
    pareto_scale: float | None
    pareto_threshold: float | None
    pareto_fit_status: str
    pareto_estimator: str = "generalized_pareto_empirical_bayes"

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
            "pareto_k_regime": _pareto_k_regime(self.pareto_k),
            "pareto_tail_count": self.pareto_tail_count,
            "pareto_scale": self.pareto_scale,
            "pareto_threshold": self.pareto_threshold,
            "pareto_fit_status": self.pareto_fit_status,
            "pareto_estimator": self.pareto_estimator,
            "pareto_k_ci_lower": None,
            "pareto_k_ci_upper": None,
            "pareto_k_ci_method": _PSIS_K_CI_METHOD,
        }

    def pareto_fit_metadata(self) -> dict[str, object]:
        """Return the complete GPD diagnostic fit metadata."""

        return {
            "shape_k": self.pareto_k,
            "scale": self.pareto_scale,
            "threshold": self.pareto_threshold,
            "tail_count": self.pareto_tail_count,
            "fit_status": self.pareto_fit_status,
            "estimator": self.pareto_estimator,
            "k_regime": _pareto_k_regime(self.pareto_k),
            "k_ci_lower": None,
            "k_ci_upper": None,
            "k_ci_method": _PSIS_K_CI_METHOD,
        }


def _state_qc_thresholds(config: ReweightingQCConfig) -> dict[str, object]:
    return {
        "min_frame_count": config.min_frame_count,
        "method": config.method,
        "weight_treatment": config.weight_treatment,
        "exclusion_scope": config.exclusion_scope,
        "exclusion_strategy": config.exclusion_strategy,
        "pareto_k_threshold": config.pareto_k_threshold,
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
            pareto_scale=None,
            pareto_threshold=None,
            pareto_fit_status="no_finite_positive_weights",
            pareto_estimator="arviz_gpd",
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

    from .psis import gpd_psis_log_weights

    values = np.asarray([frame_reweighting_delta(frame) for frame in frames])
    _, fit = gpd_psis_log_weights(-beta * (values - values.min()))
    pareto_fit = _ParetoTailFit(
        shape_k=fit["psis_pareto_k"], scale=None, threshold=None,
        tail_count=fit["psis_tail_count"], status=fit["psis_fit_status"],
    )
    pareto_k = pareto_fit.shape_k
    pareto_tail_count = pareto_fit.tail_count
    pareto_k_threshold = config.pareto_k_threshold
    reasons: list[str] = []
    if config.policy != "none" and pareto_k is not None and pareto_k >= pareto_k_threshold:
        reasons.append(
            f"Pareto tail shape fails the fixed reporting cutoff "
            f"({pareto_k:.3g} >= {pareto_k_threshold:.3g})"
        )

    return _StateQCReport(
        status="flagged" if reasons else "ok",
        reasons=tuple(reasons),
        frame_count=frame_count,
        effective_sample_size=ess,
        effective_sample_fraction=ess / frame_count if frame_count else 0.0,
        max_normalized_weight=max_weight,
        dominant_frame_index=dominant_frame.frame_index,
        dominant_frame_delta_u_kj_mol=float(frame_reweighting_delta(dominant_frame)),
        dominant_frame_weight_fraction=max_weight,
        leave_one_out_delta_g_shift_kj_mol=leave_one_out_shift,
        pareto_k=pareto_k,
        pareto_k_threshold=pareto_k_threshold,
        pareto_tail_count=pareto_tail_count,
        pareto_scale=pareto_fit.scale,
        pareto_threshold=pareto_fit.threshold,
        pareto_fit_status=pareto_fit.status,
        pareto_estimator="arviz_gpd",
    )


def _qc_state_task(
    task: tuple[str, list[FrameResult], float, ReweightingQCConfig],
) -> tuple[str, list[FrameResult], dict[str, object]]:
    """Record state diagnostics without excluding any additional frames."""

    state_id, original_state_frames, beta, qc_config = task
    report = _state_qc_report(original_state_frames, beta=beta, config=qc_config)
    return (
        state_id,
        original_state_frames,
        {
            "original_qc_report": report,
            "final_qc_report": report,
            "excluded_frame_indices": (),
            "original_frame_count": len(original_state_frames),
        },
    )


@dataclass
class ReweightingAggregator:
    """Aggregate per-frame deltaU values into reweighted microstate populations."""

    temperature_kelvin: float
    qc_config: ReweightingQCConfig = field(default_factory=ReweightingQCConfig)
    workers: int = 1

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
        reweighting_config: ReweightingConfig | None = None,
    ) -> list[MicrostateResult]:
        """Compute Zwanzig reweighting factors and reconstruct target state populations."""

        from felp.core import ReweightingConfig
        from felp.reweighting.orchestrator import ReweightingOrchestrator

        frame_results = list(frame_results)
        if not frame_results:
            return []

        config = reweighting_config or ReweightingConfig(methods=("independent",))
        outcome = ReweightingOrchestrator(
            temperature_kelvin=self.temperature_kelvin,
            config=config,
            qc_config=self.qc_config,
            workers=self.workers,
        ).run_single(
            frame_results,
            atlas,
            reference_counts=reference_counts,
            sampling_metadata=sampling_metadata,
        )
        return outcome.microstate_results
