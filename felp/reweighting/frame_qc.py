"""Global frame-level QC for partition-independent outlier screening."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from felp.core import FrameResult, ReweightingQCConfig, frame_reweighting_delta
from felp.reweighting.aggregator import (
    _effective_sample_size,
    _state_local_weights,
    _state_qc_report,
)


@dataclass(frozen=True)
class FrameQCResult:
    """Outcome of global adaptive QC filtering."""

    frames: list[FrameResult]
    weights: np.ndarray
    excluded_frame_indices: tuple[int, ...]
    excluded_frame_delta_u_kj_mol: tuple[float, ...]
    global_ess: float
    global_ess_fraction: float
    max_normalized_weight: float
    status: str
    original_pareto_k: float | None
    final_pareto_k: float | None
    pareto_k_threshold: float | None
    original_pareto_fit: dict[str, object]
    final_pareto_fit: dict[str, object]
    original_ess: float
    original_max_normalized_weight: float
    original_leave_one_out_shift_kj_mol: float | None
    final_leave_one_out_shift_kj_mol: float | None
    dominant_frame_index: int | None
    dominant_frame_delta_u_kj_mol: float | None
    dominant_frame_weight_fraction: float
    stop_reason: str
    reliability_unresolved: bool
    cap_exhausted: bool
    exclusion_strategy: str
    exclusion_history: tuple[dict[str, Any], ...]
    reasons: tuple[str, ...]

    def with_weights(
        self,
        weights: np.ndarray,
        *,
        frames: list[FrameResult] | None = None,
    ) -> "FrameQCResult":
        """Return a copy whose SNIS weights have been replaced after QC."""

        remaining_frames = self.frames if frames is None else frames
        remaining_weights = np.asarray(weights, dtype=float)
        ess = _effective_sample_size(remaining_weights)
        weight_sum = float(remaining_weights.sum())
        return replace(
            self,
            frames=remaining_frames,
            weights=remaining_weights,
            global_ess=ess,
            global_ess_fraction=ess / len(remaining_frames) if remaining_frames else 0.0,
            max_normalized_weight=(
                float(remaining_weights.max() / weight_sum)
                if remaining_weights.size and weight_sum > 0.0
                else 0.0
            ),
        )

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-serializable global-QC metadata."""

        return {
            "status": self.status,
            "exclusion_strategy": self.exclusion_strategy,
            "excluded_frame_indices": list(self.excluded_frame_indices),
            "excluded_frame_delta_u_kj_mol": list(self.excluded_frame_delta_u_kj_mol),
            "excluded_frame_count": len(self.excluded_frame_indices),
            "exclusion_history": [dict(item) for item in self.exclusion_history],
            "global_ess": self.global_ess,
            "global_ess_fraction": self.global_ess_fraction,
            "max_normalized_weight": self.max_normalized_weight,
            "original_pareto_k": self.original_pareto_k,
            "final_pareto_k": self.final_pareto_k,
            "pareto_k_threshold": self.pareto_k_threshold,
            "original_pareto_fit": dict(self.original_pareto_fit),
            "final_pareto_fit": dict(self.final_pareto_fit),
            "original_ess": self.original_ess,
            "original_max_normalized_weight": self.original_max_normalized_weight,
            "original_leave_one_out_shift_kj_mol": (
                self.original_leave_one_out_shift_kj_mol
            ),
            "final_leave_one_out_shift_kj_mol": self.final_leave_one_out_shift_kj_mol,
            "dominant_frame_index": self.dominant_frame_index,
            "dominant_frame_delta_u_kj_mol": self.dominant_frame_delta_u_kj_mol,
            "dominant_frame_weight_fraction": self.dominant_frame_weight_fraction,
            "stop_reason": self.stop_reason,
            "reliability_unresolved": self.reliability_unresolved,
            "cap_exhausted": self.cap_exhausted,
            "reasons": list(self.reasons),
        }


def apply_global_frame_qc(
    frames: list[FrameResult],
    weights: np.ndarray | None = None,
    *,
    qc_config: ReweightingQCConfig,
    beta: float | None = None,
) -> FrameQCResult:
    """Apply the global tail cut before state grouping; PSIS follows separately."""

    remaining_frames = list(frames)
    if weights is None:
        remaining_weights, _ = _state_local_weights(remaining_frames, float(beta or 1.0))
    else:
        remaining_weights = np.asarray(weights, dtype=float)
        if remaining_weights.size != len(remaining_frames):
            raise ValueError("Global QC weights must align with the frame list.")

    report_beta = 1.0 if beta is None else float(beta)
    original_report = _state_qc_report(
        remaining_frames, beta=report_beta, config=qc_config
    )
    excluded_indices: list[int] = []
    excluded_delta_u: list[float] = []
    history: list[dict[str, Any]] = []
    stop_reason = "no_exclusion_requested"

    if qc_config.policy == "exclude-frames":
        from .global_gpd import first_passing_global_gpd_cut
        delta = np.asarray([frame_reweighting_delta(frame) for frame in frames])
        keep, removed, history, stop_reason = first_passing_global_gpd_cut(
            delta, beta=report_beta, cutoff=qc_config.pareto_k_threshold,
            min_retained=max(20, qc_config.min_frame_count),
        )
        remaining_frames = [frame for frame, retained in zip(frames, keep) if retained]
        excluded_indices = [int(frames[i].frame_index) for i in removed]
        excluded_delta_u = [float(delta[i]) for i in removed]
        # Rescale after trimming so an extreme original minimum cannot make
        # every retained raw weight underflow before its ESS is calculated.
        remaining_weights, _ = _state_local_weights(remaining_frames, report_beta)
    final_report = _state_qc_report(
        remaining_frames, beta=report_beta, config=qc_config
    )
    weight_sum = float(remaining_weights.sum())
    normalized = (
        remaining_weights / weight_sum if weight_sum > 0.0 else remaining_weights
    )
    global_ess = _effective_sample_size(remaining_weights)
    frame_count = len(remaining_frames)
    final_k_fails = (
        final_report.pareto_k is not None
        and final_report.pareto_k_threshold is not None
        and final_report.pareto_k >= final_report.pareto_k_threshold
    )
    reliability_unresolved = final_k_fails or stop_reason in {"empty", "global_gpd_k_unresolved"}
    if stop_reason == "global_gpd_k_passed" and (
        final_report.pareto_k is None or final_report.pareto_k >= qc_config.pareto_k_threshold
    ):
        raise ValueError("Optimized global GPD cut disagrees with the full ArviZ diagnostic")
    if excluded_indices:
        status = "filtered"
    elif original_report.status == "flagged" or reliability_unresolved:
        status = "flagged"
    else:
        status = "ok"
    return FrameQCResult(
        frames=remaining_frames,
        weights=remaining_weights,
        excluded_frame_indices=tuple(excluded_indices),
        excluded_frame_delta_u_kj_mol=tuple(excluded_delta_u),
        global_ess=global_ess,
        global_ess_fraction=global_ess / frame_count if frame_count else 0.0,
        max_normalized_weight=float(normalized.max()) if normalized.size else 0.0,
        status=status,
        original_pareto_k=original_report.pareto_k,
        final_pareto_k=final_report.pareto_k,
        pareto_k_threshold=final_report.pareto_k_threshold,
        original_pareto_fit=original_report.pareto_fit_metadata(),
        final_pareto_fit=final_report.pareto_fit_metadata(),
        original_ess=original_report.effective_sample_size,
        original_max_normalized_weight=original_report.max_normalized_weight,
        original_leave_one_out_shift_kj_mol=(
            original_report.leave_one_out_delta_g_shift_kj_mol
        ),
        final_leave_one_out_shift_kj_mol=final_report.leave_one_out_delta_g_shift_kj_mol,
        dominant_frame_index=final_report.dominant_frame_index,
        dominant_frame_delta_u_kj_mol=final_report.dominant_frame_delta_u_kj_mol,
        dominant_frame_weight_fraction=final_report.dominant_frame_weight_fraction,
        stop_reason=stop_reason,
        reliability_unresolved=reliability_unresolved,
        cap_exhausted=False,
        exclusion_strategy=qc_config.exclusion_strategy,
        exclusion_history=tuple(history),
        reasons=tuple(dict.fromkeys((*original_report.reasons, *final_report.reasons))),
    )
