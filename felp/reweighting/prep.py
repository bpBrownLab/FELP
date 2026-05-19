"""Shared preparation for modular reweighting estimators."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from felp.core import FrameResult, ReweightingQCConfig, ReweightingError
from .aggregator import (
    _effective_sample_size,
    _qc_state_task,
    _reference_probability_map,
    _state_local_weights,
    _unsupported_positive_prior_states,
    is_reweighting_eligible_frame,
    psis_smooth_weights,
)
from .atlas import MicrostateAtlas
from .topology import GridTopology, build_grid_topology


@dataclass
class StatePrepData:
    """Per-state inputs produced by shared preparation."""

    state_id: str
    frames: list[FrameResult]
    original_frame_count: int
    assigned_frame_count: int
    reference_probability: float
    local_factor: float
    local_ess: float
    treated_weights: np.ndarray
    mean_delta_u_kj_mol: float | None
    std_delta_u_kj_mol: float | None
    mean_boltzmann_factor: float | None
    std_boltzmann_factor: float | None
    original_qc_report: Any
    final_qc_report: Any
    excluded_frame_indices: tuple[int, ...]
    psis_weight_metadata: dict[str, object]


@dataclass
class ReweightingPrep:
    """Shared inputs for all microstate reweighting backends."""

    atlas: MicrostateAtlas
    reference_probabilities: dict[str, float]
    reference_probability_source: str
    grouped_original: dict[str, list[FrameResult]]
    filtered_grouped: dict[str, list[FrameResult]]
    qc_by_state: dict[str, dict[str, object]]
    state_data: dict[str, StatePrepData]
    global_delta_shift: float
    eligible_frame_count: int
    filtered_eligible_frame_count: int
    reference_counts: dict[str, int] | None
    topology: GridTopology | None = None
    beta: float = 0.0
    temperature_kelvin: float = 300.0
    qc_config: ReweightingQCConfig = field(default_factory=ReweightingQCConfig)


def build_reweighting_prep(
    frame_results: list[FrameResult],
    atlas: MicrostateAtlas,
    *,
    beta: float,
    temperature_kelvin: float,
    qc_config: ReweightingQCConfig,
    reference_counts: dict[str, int] | None = None,
    workers: int = 1,
    max_neighbors: int = 4,
) -> ReweightingPrep:
    """Run grouping, priors, QC, and per-state weight construction once."""

    eligible_frames = [
        result for result in frame_results if is_reweighting_eligible_frame(result)
    ]
    if not eligible_frames:
        raise ReweightingError("No eligible frames available for reweighting.")

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
    qc_tasks = [
        (
            definition.state_id,
            list(grouped.get(definition.state_id, [])),
            beta,
            qc_config,
        )
        for definition in atlas.definitions
    ]
    worker_count = max(1, int(workers))
    if worker_count > 1 and len(qc_tasks) > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            qc_results = list(executor.map(_qc_state_task, qc_tasks))
    else:
        qc_results = [_qc_state_task(task) for task in qc_tasks]
    for state_id, filtered_state_frames, qc_state in qc_results:
        filtered_grouped[state_id] = filtered_state_frames
        qc_by_state[state_id] = qc_state

    filtered_eligible_frames = [
        frame
        for definition in atlas.definitions
        for frame in filtered_grouped.get(definition.state_id, [])
    ]
    if not filtered_eligible_frames:
        raise ReweightingError(
            "No eligible frames remain after reweighting QC filtering."
        )

    global_delta_shift = min(
        result.delta_u_kj_mol for result in filtered_eligible_frames
    )
    state_data: dict[str, StatePrepData] = {}
    for definition in atlas.definitions:
        state_id = definition.state_id
        state_frames = filtered_grouped.get(state_id, [])
        qc_state = qc_by_state[state_id]
        original_qc_report = qc_state["original_qc_report"]
        final_qc_report = qc_state["final_qc_report"]
        excluded_frame_indices = qc_state["excluded_frame_indices"]
        original_frame_count = int(qc_state["original_frame_count"])
        assigned_frame_count = (
            int(reference_counts.get(state_id, 0))
            if reference_counts is not None
            else original_frame_count
        )
        reference_probability = reference_probabilities.get(state_id, 0.0)

        if state_frames:
            delta_values = np.asarray(
                [frame.delta_u_kj_mol for frame in state_frames],
                dtype=float,
            )
            local_weights, local_delta_shift = _state_local_weights(
                state_frames,
                beta,
            )
            psis_metadata: dict[str, object] = {
                "psis_smoothed": False,
                "psis_smoothed_frame_count": 0,
                "psis_pareto_k": None,
                "psis_tail_count": 0,
                "psis_truncation_cap": None,
            }
            if qc_config.weight_treatment == "psis-smoothed":
                local_weights, psis_metadata = psis_smooth_weights(local_weights)
            local_scale = (
                math.exp(-beta * (local_delta_shift - global_delta_shift))
                if local_delta_shift is not None
                else 0.0
            )
            treated_weights = local_scale * local_weights
            local_factor = float(treated_weights.mean())
            local_ess = _effective_sample_size(treated_weights)
            mean_delta_u = float(delta_values.mean())
            std_delta_u = float(delta_values.std(ddof=0))
            mean_weight = float(treated_weights.mean())
            std_weight = float(treated_weights.std(ddof=0))
        else:
            treated_weights = np.asarray([], dtype=float)
            local_factor = 0.0
            local_ess = 0.0
            mean_delta_u = None
            std_delta_u = None
            mean_weight = None
            std_weight = None
            psis_metadata = {
                "psis_smoothed": False,
                "psis_smoothed_frame_count": 0,
                "psis_pareto_k": None,
                "psis_tail_count": 0,
                "psis_truncation_cap": None,
            }

        state_data[state_id] = StatePrepData(
            state_id=state_id,
            frames=state_frames,
            original_frame_count=original_frame_count,
            assigned_frame_count=assigned_frame_count,
            reference_probability=reference_probability,
            local_factor=local_factor,
            local_ess=local_ess,
            treated_weights=treated_weights,
            mean_delta_u_kj_mol=mean_delta_u,
            std_delta_u_kj_mol=std_delta_u,
            mean_boltzmann_factor=mean_weight,
            std_boltzmann_factor=std_weight,
            original_qc_report=original_qc_report,
            final_qc_report=final_qc_report,
            excluded_frame_indices=excluded_frame_indices,
            psis_weight_metadata=psis_metadata,
        )

    topology = build_grid_topology(atlas, max_neighbors=max_neighbors)
    return ReweightingPrep(
        atlas=atlas,
        reference_probabilities=reference_probabilities,
        reference_probability_source=reference_probability_source,
        grouped_original=dict(grouped),
        filtered_grouped=filtered_grouped,
        qc_by_state=qc_by_state,
        state_data=state_data,
        global_delta_shift=global_delta_shift,
        eligible_frame_count=len(eligible_frames),
        filtered_eligible_frame_count=len(filtered_eligible_frames),
        reference_counts=reference_counts,
        topology=topology,
        beta=beta,
        temperature_kelvin=temperature_kelvin,
        qc_config=qc_config,
    )
