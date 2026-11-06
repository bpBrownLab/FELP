"""Neighbor-pooled Zwanzig reweighting on a CV grid.

See docs/methods.md for the blend estimator and L-curve tau selection.
"""

from __future__ import annotations

import math

import numpy as np

from felp.core import NeighborPoolConfig, ReweightingConfig
from felp.reweighting.engines._finalize import finalize_microstate_outcome
from felp.reweighting.frame_weights import frame_weights_from_prep
from felp.reweighting.outcome import ReweightingOutcome
from felp.reweighting.prep import ReweightingPrep

_ESTIMATOR_BLEND = (
    "pi_target(i) proportional to pi_reference(i) * f_i with "
    "ESS-weighted partial pooling of local Boltzmann factors toward CV neighbors"
)


def _topology_metadata(prep: ReweightingPrep) -> dict[str, object]:
    metadata: dict[str, object] = {
        "topology_source": prep.topology.source if prep.topology is not None else None,
        "topology_warnings": (
            list(prep.topology.warnings) if prep.topology is not None else []
        ),
    }
    if prep.topology is not None and prep.topology.source != "knn":
        metadata["grid_connectivity"] = prep.topology.connectivity
        if prep.topology.chebyshev_radius is not None:
            metadata["chebyshev_radius"] = prep.topology.chebyshev_radius
        metadata["periodic_cv_names"] = list(prep.topology.periodic_cv_names)
    return metadata


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


def _blend_tau_candidate_grid(positive_ess: list[float], *, num: int = 30) -> np.ndarray:
    tau_min = 1.0
    tau_max = max(max(positive_ess) * 10.0, tau_min * 10.0)
    return np.geomspace(tau_min, tau_max, num=num)


def _blend_misfit_and_roughness(
    prep: ReweightingPrep,
    *,
    tau: float,
    local_factors: dict[str, float],
    local_ess: dict[str, float],
    min_local_weight: float,
) -> tuple[float, float]:
    blended = _blended_factors_for_tau(
        tau,
        prep,
        local_factors=local_factors,
        local_ess=local_ess,
        min_local_weight=min_local_weight,
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
    return float(misfit), float(roughness)


def _blend_tau_scan(
    prep: ReweightingPrep,
    *,
    local_factors: dict[str, float],
    local_ess: dict[str, float],
    pool_config: NeighborPoolConfig,
) -> list[dict[str, float]]:
    positive_ess = [ess for ess in local_ess.values() if ess > 0.0]
    scan: list[dict[str, float]] = []
    for tau in _blend_tau_candidate_grid(positive_ess):
        misfit, roughness = _blend_misfit_and_roughness(
            prep,
            tau=float(tau),
            local_factors=local_factors,
            local_ess=local_ess,
            min_local_weight=pool_config.min_local_weight,
        )
        scan.append(
            {
                "tau": float(tau),
                "misfit": misfit,
                "roughness": roughness,
            }
        )
    return scan


def _lcurve_corner_index(misfit: np.ndarray, roughness: np.ndarray) -> int:
    """Index of the L-curve corner via max distance to the chord in log-log space."""

    size = len(misfit)
    if size == 0:
        return 0
    if size < 3:
        return size // 2

    log_misfit = np.log10(np.maximum(misfit, 1.0e-30))
    log_roughness = np.log10(np.maximum(roughness, 1.0e-30))
    x0, y0 = float(log_misfit[0]), float(log_roughness[0])
    x1, y1 = float(log_misfit[-1]), float(log_roughness[-1])
    dx = x1 - x0
    dy = y1 - y0
    denom = math.hypot(dx, dy)
    if denom <= 1.0e-12:
        return size // 2

    best_index = 0
    best_distance = -1.0
    for index in range(size):
        distance = abs(
            dy * (float(log_misfit[index]) - x0) - dx * (float(log_roughness[index]) - y0)
        ) / denom
        if distance > best_distance:
            best_distance = distance
            best_index = index
    return best_index


def _lcurve_tau(
    prep: ReweightingPrep,
    *,
    local_factors: dict[str, float],
    local_ess: dict[str, float],
    pool_config: NeighborPoolConfig,
) -> tuple[float, dict[str, object]]:
    """Pick tau at the corner of the misfit vs roughness L-curve."""

    scan = _blend_tau_scan(
        prep,
        local_factors=local_factors,
        local_ess=local_ess,
        pool_config=pool_config,
    )
    misfit = np.asarray([entry["misfit"] for entry in scan], dtype=float)
    roughness = np.asarray([entry["roughness"] for entry in scan], dtype=float)
    corner_index = _lcurve_corner_index(misfit, roughness)
    corner = scan[corner_index]
    return float(corner["tau"]), {
        "lcurve_corner_index": corner_index,
        "lcurve_corner_misfit": float(corner["misfit"]),
        "lcurve_corner_roughness": float(corner["roughness"]),
        "lcurve_tau_candidates_evaluated": len(scan),
        "lcurve_scan": scan,
    }


def _resolve_tau_blend(
    prep: ReweightingPrep,
    *,
    local_factors: dict[str, float],
    local_ess: dict[str, float],
    pool_config: NeighborPoolConfig,
) -> tuple[float, dict[str, object]]:
    if pool_config.tau is not None:
        return float(pool_config.tau), {"tau_mode": "fixed", "tau": float(pool_config.tau)}

    positive_ess = [ess for ess in local_ess.values() if ess > 0.0]
    if not positive_ess:
        return 10.0, {"tau_mode": pool_config.tau_mode, "tau": 10.0}


    tau, lcurve_metadata = _lcurve_tau(
        prep,
        local_factors=local_factors,
        local_ess=local_ess,
        pool_config=pool_config,
    )
    return tau, {"tau_mode": "lcurve", "tau": tau, **lcurve_metadata}


def _aggregate_blend(
    prep: ReweightingPrep,
    *,
    pool_config: NeighborPoolConfig,
    local_factors: dict[str, float],
    local_ess: dict[str, float],
    sampling_metadata: dict[str, object] | None,
) -> ReweightingOutcome:
    tau, tau_metadata = _resolve_tau_blend(
        prep,
        local_factors=local_factors,
        local_ess=local_ess,
        pool_config=pool_config,
    )
    blended_factors = _blended_factors_for_tau(
        tau,
        prep,
        local_factors=local_factors,
        local_ess=local_ess,
        min_local_weight=pool_config.min_local_weight,
    )

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
            "pool_solver": "blend",
            **tau_metadata,
            **_topology_metadata(prep),
            "high_pooling_low_local_ess_state_ids": high_pooling_low_ess[:20],
        }
    }
    if prep.topology is None:
        method_metadata["neighbor_pool"]["warning"] = (
            "No grid topology available; neighbor pooling degraded to local factors."
        )

    outcome = finalize_microstate_outcome(
        prep,
        method_id="neighbor_pool",
        state_reweighting_estimator=_ESTIMATOR_BLEND,
        reweight_factors=blended_factors,
        per_state_metadata=per_state_metadata,
        sampling_metadata=sampling_metadata,
        method_metadata=method_metadata,
    )
    frame_weights = frame_weights_from_prep(prep, blended_factors)
    if frame_weights:
        outcome.auxiliary = {"frame_weights": frame_weights}
    return outcome


class NeighborPoolEngine:
    """Neighbor-pooled Zwanzig reweighting on a CV grid."""

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
        return _aggregate_blend(
            prep,
            pool_config=pool_config,
            local_factors=local_factors,
            local_ess=local_ess,
            sampling_metadata=sampling_metadata,
        )
