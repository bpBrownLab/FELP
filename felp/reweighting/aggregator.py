"""Local reweighting aggregation for FELP."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from felp.core import FrameResult, K_BOLTZMANN_KJ_PER_MOL_K, MicrostateResult
from .atlas import MicrostateAtlas


def is_reweighting_eligible_frame(result: FrameResult) -> bool:
    if result.metadata.get("status") == "failed":
        return False
    if not (
        math.isfinite(result.reference_energy_kj_mol)
        and math.isfinite(result.perturbed_energy_kj_mol)
        and math.isfinite(result.delta_u_kj_mol)
    ):
        return False
    return True


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

        empirical_total = len(eligible_frames)
        reference_probabilities: dict[str, float] = {}

        use_loaded = atlas.use_loaded_reference_probabilities and all(
            definition.reference_probability is not None for definition in atlas.definitions
        )
        if use_loaded:
            normalization = sum(
                float(definition.reference_probability) for definition in atlas.definitions
            )
            for definition in atlas.definitions:
                reference_probabilities[definition.state_id] = (
                    float(definition.reference_probability) / normalization
                )
        else:
            for definition in atlas.definitions:
                reference_probabilities[definition.state_id] = len(
                    grouped.get(definition.state_id, [])
                ) / empirical_total

        preliminary_results = []
        for definition in atlas.definitions:
            state_frames = grouped.get(definition.state_id, [])
            if state_frames:
                delta_values = np.array(
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

            reference_probability = reference_probabilities.get(definition.state_id, 0.0)
            preliminary_results.append(
                {
                    "definition": definition,
                    "frame_count": len(state_frames),
                    "reference_probability": reference_probability,
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
            normalization = 1.0

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
                        "reference_probability_source": (
                            "definition"
                            if use_loaded
                            else "empirical"
                        ),
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
