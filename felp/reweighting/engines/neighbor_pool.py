"""Neighbor-pooled Zwanzig reweighting on a CV grid."""

from __future__ import annotations

import math

import numpy as np

from felp.core import NeighborPoolConfig, ReweightingConfig
from felp.reweighting.engines._finalize import finalize_microstate_outcome
from felp.reweighting.outcome import ReweightingOutcome
from felp.reweighting.prep import ReweightingPrep

_ESTIMATOR = (
    "pi_target(i) proportional to pi_reference(i) * f_i with "
    "ESS-weighted partial pooling of local Boltzmann factors toward CV neighbors"
)


def _pool_factor(
    state_id: str,
    prep: ReweightingPrep,
    *,
    local_factors: dict[str, float],
    local_ess: dict[str, float],
) -> tuple[float, float, tuple[str, ...]]:
    topology = prep.topology
    if topology is None:
        return local_factors[state_id], 1.0, ()

    neighbor_ids = topology.neighbor_ids(state_id)
    if not neighbor_ids:
        return local_factors[state_id], 1.0, ()

    weights = []
    factors = []
    for neighbor_id in neighbor_ids:
        ess = local_ess.get(neighbor_id, 0.0)
        factor = local_factors.get(neighbor_id, 0.0)
        if ess <= 0.0 or factor <= 0.0:
            continue
        weights.append(ess)
        factors.append(factor)
    if not weights:
        return local_factors[state_id], 1.0, neighbor_ids

    pooled = sum(weight * factor for weight, factor in zip(weights, factors)) / sum(
        weights
    )
    return pooled, 0.0, neighbor_ids


def _blended_factors_for_tau(
    tau: float,
    prep: ReweightingPrep,
    *,
    local_factors: dict[str, float],
    local_ess: dict[str, float],
    min_local_weight: float,
) -> dict[str, float]:
    blended: dict[str, float] = {}
    for state_id, state in prep.state_data.items():
        raw_factor = local_factors[state_id]
        pooled_factor, _, _ = _pool_factor(
            state_id,
            prep,
            local_factors=local_factors,
            local_ess=local_ess,
        )
        ess = local_ess[state_id]
        pooling_weight = ess / (ess + tau) if ess + tau > 0 else 0.0
        if min_local_weight > 0.0:
            pooling_weight = max(pooling_weight, min_local_weight)
            pooling_weight = min(pooling_weight, 1.0)
        blended[state_id] = pooling_weight * raw_factor + (1.0 - pooling_weight) * pooled_factor
    return blended


def _log_factor_grid_roughness(
    prep: ReweightingPrep,
    log_factors: dict[str, float],
) -> float:
    if prep.topology is None:
        return 0.0
    roughness = 0.0
    for state_id, log_factor in log_factors.items():
        for neighbor_id in prep.topology.neighbor_ids(state_id):
            neighbor_log = log_factors.get(neighbor_id)
            if neighbor_log is None:
                continue
            diff = log_factor - neighbor_log
            roughness += diff * diff
    return 0.5 * roughness


def _resolve_tau(
    prep: ReweightingPrep,
    *,
    local_factors: dict[str, float],
    local_ess: dict[str, float],
    pool_config: NeighborPoolConfig,
) -> tuple[float, dict[str, object]]:
    """Choose pooling strength tau from config or data-driven criteria."""

    if pool_config.tau is not None:
        return float(pool_config.tau), {"tau_mode": "fixed", "tau": float(pool_config.tau)}

    positive_ess = [ess for ess in local_ess.values() if ess > 0.0]
    if not positive_ess:
        return 10.0, {"tau_mode": pool_config.tau_mode, "tau": 10.0}

    if pool_config.tau_mode == "mean_ess":
        tau = float(sum(positive_ess) / len(positive_ess))
        return tau, {"tau_mode": "mean_ess", "tau": tau}
    if pool_config.tau_mode == "median_ess":
        tau = float(np.median(positive_ess))
        return tau, {"tau_mode": "median_ess", "tau": tau}

    tau, gcv_metadata = _gcv_tau(
        prep,
        local_factors=local_factors,
        local_ess=local_ess,
        pool_config=pool_config,
    )
    return tau, {"tau_mode": "gcv", "tau": tau, **gcv_metadata}


def _gcv_tau(
    prep: ReweightingPrep,
    *,
    local_factors: dict[str, float],
    local_ess: dict[str, float],
    pool_config: NeighborPoolConfig,
) -> tuple[float, dict[str, object]]:
    """Pick tau by minimizing fit-to-local vs grid-smoothness on log factors."""

    positive_ess = [ess for ess in local_ess.values() if ess > 0.0]
    frame_counts = [
        len(prep.state_data[state_id].frames)
        for state_id in prep.state_data
        if prep.state_data[state_id].frames
    ]
    median_frames = float(np.median(frame_counts)) if frame_counts else 1.0
    tau_min = 1.0
    tau_max = max(max(positive_ess) * 10.0, tau_min * 10.0)
    candidates = np.geomspace(tau_min, tau_max, num=30)

    best_tau = float(np.median(positive_ess))
    best_score = float("inf")
    scores: list[dict[str, float]] = []
    for tau in candidates:
        blended = _blended_factors_for_tau(
            float(tau),
            prep,
            local_factors=local_factors,
            local_ess=local_ess,
            min_local_weight=pool_config.min_local_weight,
        )
        misfit = 0.0
        log_blended: dict[str, float] = {}
        for state_id, state in prep.state_data.items():
            local_factor = local_factors[state_id]
            if not state.frames or local_factor <= 0.0:
                continue
            blended_factor = blended[state_id]
            if blended_factor <= 0.0:
                continue
            log_local = math.log(local_factor)
            log_blend = math.log(blended_factor)
            log_blended[state_id] = log_blend
            misfit += state.local_ess * (log_local - log_blend) ** 2
        roughness = _log_factor_grid_roughness(prep, log_blended)
        score = misfit + median_frames * roughness
        scores.append(
            {
                "tau": float(tau),
                "score": float(score),
                "misfit": float(misfit),
                "roughness": float(roughness),
            }
        )
        if score < best_score:
            best_score = score
            best_tau = float(tau)

    return best_tau, {
        "gcv_score": best_score,
        "gcv_median_frame_weight": median_frames,
        "gcv_tau_candidates_evaluated": len(candidates),
        "gcv_best_tau": best_tau,
    }


class NeighborPoolEngine:
    """ESS-weighted partial pooling toward grid neighbors."""

    method_id = "neighbor_pool"

    def aggregate(
        self,
        prep: ReweightingPrep,
        *,
        config: ReweightingConfig,
        sampling_metadata: dict[str, object] | None = None,
    ) -> ReweightingOutcome:
        pool_config = config.neighbor_pool
        local_factors = {
            state_id: state.local_factor for state_id, state in prep.state_data.items()
        }
        local_ess = {
            state_id: state.local_ess for state_id, state in prep.state_data.items()
        }
        tau, tau_metadata = _resolve_tau(
            prep,
            local_factors=local_factors,
            local_ess=local_ess,
            pool_config=pool_config,
        )

        reweight_factors: dict[str, float] = {}
        per_state_metadata: dict[str, dict[str, object]] = {}
        high_pooling_low_ess: list[str] = []

        for state_id, state in prep.state_data.items():
            raw_factor = local_factors[state_id]
            pooled_factor, _, neighbor_ids = _pool_factor(
                state_id,
                prep,
                local_factors=local_factors,
                local_ess=local_ess,
            )
            ess = local_ess[state_id]
            pooling_weight = ess / (ess + tau) if ess + tau > 0 else 0.0
            if pool_config.min_local_weight > 0.0:
                pooling_weight = max(pooling_weight, pool_config.min_local_weight)
                pooling_weight = min(pooling_weight, 1.0)
            blended = pooling_weight * raw_factor + (1.0 - pooling_weight) * pooled_factor
            reweight_factors[state_id] = blended
            per_state_metadata[state_id] = {
                "raw_local_factor": raw_factor,
                "pooled_neighbor_factor": pooled_factor,
                "pooling_weight": pooling_weight,
                "neighbor_state_ids": list(neighbor_ids),
                "local_ess": ess,
            }
            if pooling_weight < 0.5 and ess < 10.0 and neighbor_ids:
                high_pooling_low_ess.append(state_id)

        method_metadata: dict[str, object] = {
            "neighbor_pool": {
                **tau_metadata,
                "topology_source": (
                    prep.topology.source if prep.topology is not None else None
                ),
                "topology_warnings": (
                    list(prep.topology.warnings) if prep.topology is not None else []
                ),
                "high_pooling_low_local_ess_state_ids": high_pooling_low_ess[:20],
            }
        }
        if prep.topology is None:
            method_metadata["neighbor_pool"]["warning"] = (
                "No grid topology available; neighbor pooling degraded to local factors."
            )

        return finalize_microstate_outcome(
            prep,
            method_id=self.method_id,
            state_reweighting_estimator=_ESTIMATOR,
            reweight_factors=reweight_factors,
            per_state_metadata=per_state_metadata,
            sampling_metadata=sampling_metadata,
            method_metadata=method_metadata,
        )
