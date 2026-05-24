"""Laplacian-regularized MAP excess free-energy reweighting."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from felp.core import K_BOLTZMANN_KJ_PER_MOL_K, LaplacianConfig, ReweightingConfig, ReweightingError
from felp.reweighting.aggregator import _state_local_weights
from felp.reweighting.engines._finalize import finalize_microstate_outcome
from felp.reweighting.frame_weights import frame_weights_from_prep
from felp.reweighting.outcome import ReweightingOutcome
from felp.reweighting.prep import ReweightingPrep

_ESTIMATOR = (
    "pi_target(i) proportional to pi_reference(i) * exp(-beta * DeltaG_i) with "
    "MAP DeltaG regularized by a grid Laplacian prior"
)


def _build_laplacian_matrix(state_ids: list[str], prep: ReweightingPrep) -> np.ndarray:
    size = len(state_ids)
    index = {state_id: position for position, state_id in enumerate(state_ids)}
    matrix = np.zeros((size, size), dtype=float)
    if prep.topology is None:
        return matrix
    for state_id in state_ids:
        position = index[state_id]
        for neighbor_id in prep.topology.neighbor_ids(state_id):
            if neighbor_id not in index:
                continue
            neighbor_position = index[neighbor_id]
            matrix[position, position] += 1.0
            matrix[position, neighbor_position] -= 1.0
    return matrix


def _laplacian_has_edges(laplacian: np.ndarray) -> bool:
    """Return whether the graph Laplacian has at least one neighbor edge."""

    return bool(np.any(laplacian < 0.0))


def _optimize_delta_g(
    prep: ReweightingPrep,
    *,
    lambda_value: float,
    max_iterations: int,
) -> tuple[dict[str, float], dict[str, object]]:
    active_states = [
        definition.state_id
        for definition in prep.atlas.definitions
        if prep.state_data[definition.state_id].frames
    ]
    if not active_states:
        raise ReweightingError("Laplacian reweighting requires at least one state with frames.")

    beta = prep.beta
    kbt = K_BOLTZMANN_KJ_PER_MOL_K * prep.temperature_kelvin
    state_ids = active_states
    index = {state_id: position for position, state_id in enumerate(state_ids)}
    laplacian = _build_laplacian_matrix(state_ids, prep)

    raw_delta_g = {}
    frame_weights_by_state: dict[str, np.ndarray] = {}
    frame_delta_by_state: dict[str, np.ndarray] = {}
    for state_id in state_ids:
        frames = prep.state_data[state_id].frames
        weights, _ = _state_local_weights(frames, beta)
        frame_weights_by_state[state_id] = weights
        frame_delta_by_state[state_id] = np.asarray(
            [frame.delta_u_kj_mol for frame in frames],
            dtype=float,
        )
        local_factor = prep.state_data[state_id].local_factor
        if local_factor > 0.0:
            raw_delta_g[state_id] = -kbt * math.log(local_factor)
        else:
            raw_delta_g[state_id] = 0.0

    if lambda_value <= 0.0 or not _laplacian_has_edges(laplacian):
        return raw_delta_g, {
            "solver": "analytic_independent_limit",
            "iterations": 0,
            "success": True,
            "laplacian_penalty": 0.0,
            "laplacian_edge_count": int(np.sum(laplacian < 0.0)),
        }

    initial = np.asarray([raw_delta_g[state_id] for state_id in state_ids], dtype=float)
    bound_radius = max(50.0, 5.0 * float(np.std(initial)) + 20.0)
    bounds = [
        (float(raw_delta_g[state_id]) - bound_radius, float(raw_delta_g[state_id]) + bound_radius)
        for state_id in state_ids
    ]

    def objective(delta_g: np.ndarray) -> float:
        """Negative MAP objective: -sum_i n_i log <exp(-beta(du-G_i))>_i + Laplacian penalty."""

        total = 0.0
        for state_id in state_ids:
            weights = np.exp(-beta * (frame_delta_by_state[state_id] - delta_g[index[state_id]]))
            log_mean = float(np.log(np.maximum(weights.mean(), 1.0e-300)))
            total -= len(frame_weights_by_state[state_id]) * log_mean
        if lambda_value > 0.0:
            total += 0.5 * lambda_value * float(delta_g @ laplacian @ delta_g)
        return total

    def gradient(delta_g: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(delta_g)
        for state_id in state_ids:
            position = index[state_id]
            count = len(frame_delta_by_state[state_id])
            grad[position] -= count * beta
        if lambda_value > 0.0:
            grad += lambda_value * (laplacian @ delta_g)
        return grad

    optimized, solver_info = _minimize(
        objective,
        gradient,
        initial,
        max_iterations=max_iterations,
        bounds=bounds,
    )
    optimized_map = {
        state_id: float(optimized[index[state_id]]) for state_id in state_ids
    }
    penalty = float(optimized @ laplacian @ optimized) if lambda_value > 0.0 else 0.0
    return optimized_map, {
        **solver_info,
        "laplacian_penalty": penalty,
        "laplacian_edge_count": int(np.sum(laplacian < 0.0)),
    }


def _minimize(
    objective: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray],
    initial: np.ndarray,
    *,
    max_iterations: int,
    bounds: list[tuple[float, float]] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    try:
        from scipy.optimize import minimize

        result = minimize(
            objective,
            initial,
            jac=gradient,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iterations},
        )
        candidate = np.asarray(result.x, dtype=float)
        if np.isfinite(candidate).all():
            return candidate, {
                "solver": "scipy_lbfgsb",
                "iterations": int(result.nit),
                "success": bool(result.success),
            }
    except ImportError:
        pass

    current = initial.copy()
    if bounds is not None:
        lower = np.asarray([item[0] for item in bounds], dtype=float)
        upper = np.asarray([item[1] for item in bounds], dtype=float)
        current = np.clip(current, lower, upper)
    step = 1.0
    iterations = 0
    for _ in range(max_iterations):
        grad = gradient(current)
        norm = float(np.linalg.norm(grad))
        if norm < 1.0e-8:
            break
        trial = current - step * grad / norm
        if bounds is not None:
            trial = np.clip(trial, lower, upper)
        if objective(trial) < objective(current):
            current = trial
            step = min(step * 1.2, 5.0)
        else:
            step *= 0.5
            if step < 1.0e-8:
                break
        iterations += 1
    return current, {
        "solver": "gradient_descent",
        "iterations": iterations,
        "success": True,
    }


class LaplacianRegularizedEngine:
    """MAP DeltaG with Laplacian smoothness on a CV grid."""

    method_id = "laplacian"

    def aggregate(
        self,
        prep: ReweightingPrep,
        *,
        config: ReweightingConfig,
        sampling_metadata: dict[str, object] | None = None,
    ) -> ReweightingOutcome:
        lap_config = config.laplacian
        kbt = K_BOLTZMANN_KJ_PER_MOL_K * prep.temperature_kelvin
        lambda_value = float(lap_config.effective_lambda_kj_mol2(kbt))

        if lambda_value > 0.0 and prep.topology is None:
            raise ReweightingError(
                "Laplacian reweighting with lambda > 0 requires a grid topology. "
                "Use --laplacian-lambda 0 for independent-like behavior on non-grid atlases."
            )

        delta_g_map, solver_metadata = _optimize_delta_g(
            prep,
            lambda_value=lambda_value,
            max_iterations=lap_config.max_iterations,
        )
        raw_delta_g = {}
        reweight_factors: dict[str, float] = {}
        per_state_metadata: dict[str, dict[str, object]] = {}

        for definition in prep.atlas.definitions:
            state_id = definition.state_id
            state = prep.state_data[state_id]
            if state.local_factor > 0.0:
                raw_delta_g[state_id] = -kbt * math.log(state.local_factor)
            else:
                raw_delta_g[state_id] = None
            if state.frames and state_id in delta_g_map:
                regularized = float(delta_g_map[state_id])
                exponent = -prep.beta * regularized
                exponent = max(min(exponent, 700.0), -700.0)
                factor = math.exp(exponent)
            else:
                regularized = None
                factor = 0.0
            reweight_factors[state_id] = factor
            per_state_metadata[state_id] = {
                "delta_g_raw_kj_mol": raw_delta_g.get(state_id),
                "delta_g_regularized_kj_mol": regularized,
                "local_ess": state.local_ess,
            }

        roughness = 0.0
        if prep.topology is not None and lambda_value > 0.0:
            for state_id in delta_g_map:
                for neighbor_id in prep.topology.neighbor_ids(state_id):
                    if neighbor_id not in delta_g_map:
                        continue
                    diff = delta_g_map[state_id] - delta_g_map[neighbor_id]
                    roughness += diff * diff
            roughness *= 0.5

        method_metadata = {
            "laplacian": {
                "lambda_kj_mol2": lap_config.lambda_kj_mol2,
                "lambda_effective_kj_mol2": lambda_value,
                "lambda_scale_by_kbt": lap_config.lambda_scale_by_kbt,
                "lambda_interpretation": (
                    "dimensionless_neighbor_coupling_times_kBT_squared"
                    if lap_config.lambda_scale_by_kbt
                    else "kj_mol2_per_neighbor_squared_difference"
                ),
                "temperature_kelvin": prep.temperature_kelvin,
                "kbt_kj_mol": kbt,
                "roughness_kj_mol2": roughness,
                "topology_source": (
                    prep.topology.source if prep.topology is not None else None
                ),
                **solver_metadata,
            }
        }

        outcome = finalize_microstate_outcome(
            prep,
            method_id=self.method_id,
            state_reweighting_estimator=_ESTIMATOR,
            reweight_factors=reweight_factors,
            per_state_metadata=per_state_metadata,
            sampling_metadata=sampling_metadata,
            method_metadata=method_metadata,
        )
        frame_weights = frame_weights_from_prep(prep, reweight_factors)
        if frame_weights:
            outcome.auxiliary = {"frame_weights": frame_weights}
        return outcome
