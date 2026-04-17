"""Local reweighting aggregation for FELP."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from felp.core import (
    FrameResult,
    K_BOLTZMANN_KJ_PER_MOL_K,
    MicrostateResult,
    ReweightingError,
)
from .atlas import MicrostateAtlas


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
        if empirical_total <= 0:
            raise ReweightingError(
                "Raw-count prior reweighting requires at least one eligible frame."
            )
        reference_probabilities = {
            definition.state_id: len(grouped.get(definition.state_id, [])) / empirical_total
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


@dataclass
class ReweightingAggregator:
    """Aggregate per-frame deltaU values into reweighted microstate populations."""

    temperature_kelvin: float

    @property
    def beta(self) -> float:
        return 1.0 / (K_BOLTZMANN_KJ_PER_MOL_K * self.temperature_kelvin)

    def aggregate(
        self,
        frame_results: Iterable[FrameResult],
        atlas: MicrostateAtlas,
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

        delta_shift = min(result.delta_u_kj_mol for result in eligible_frames)
        grouped: dict[str, list[FrameResult]] = defaultdict(list)
        for result in eligible_frames:
            grouped[result.microstate_id].append(result)

        reference_probabilities, reference_probability_source = _reference_probability_map(
            atlas,
            grouped,
            len(eligible_frames),
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

        preliminary_results = []
        for definition in atlas.definitions:
            state_frames = grouped.get(definition.state_id, [])
            if state_frames:
                delta_values = np.asarray(
                    [frame.delta_u_kj_mol for frame in state_frames],
                    dtype=float,
                )
                weights = np.exp(-self.beta * (delta_values - delta_shift))
                reweight_factor = float(weights.mean())
                mean_delta_u = float(delta_values.mean())
                std_delta_u = float(delta_values.std(ddof=0))
                mean_weight = float(weights.mean())
                std_weight = float(weights.std(ddof=0))
                ess = float((weights.sum() ** 2) / max(np.square(weights).sum(), 1.0e-12))
            else:
                reweight_factor = 0.0
                mean_delta_u = None
                std_delta_u = None
                mean_weight = None
                std_weight = None
                ess = 0.0

            preliminary_results.append(
                {
                    "definition": definition,
                    "frame_count": len(state_frames),
                    "reference_probability": reference_probabilities.get(
                        definition.state_id, 0.0
                    ),
                    "reweight_factor": reweight_factor,
                    "effective_sample_size": ess,
                    "mean_delta_u_kj_mol": mean_delta_u,
                    "std_delta_u_kj_mol": std_delta_u,
                    "mean_boltzmann_factor": mean_weight,
                    "std_boltzmann_factor": std_weight,
                }
            )

        normalization = sum(
            item["reference_probability"] * item["reweight_factor"]
            for item in preliminary_results
        )
        if normalization <= 0.0:
            raise ReweightingError(
                "The reweighted normalization constant is non-positive. "
                "Check the chosen prior mode and frame support."
            )

        microstate_results: list[MicrostateResult] = []
        for item in preliminary_results:
            definition = item["definition"]
            target_probability = (
                item["reference_probability"] * item["reweight_factor"] / normalization
            )
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
                        "delta_shift_kj_mol": delta_shift,
                        "reference_probability_source": reference_probability_source,
                        "prior_probability_mode": atlas.reference_probability_mode,
                        "unsupported_state_count": 0,
                        "unsupported_state_ids": [],
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
