"""Frame-level SNIS reweighting in CV space."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from felp.core import K_BOLTZMANN_KJ_PER_MOL_K, ReweightingConfig
from felp.reweighting.aggregator import _effective_sample_size, is_reweighting_eligible_frame
from felp.reweighting.engines._finalize import finalize_microstate_outcome
from felp.reweighting.frame_qc import apply_global_frame_qc
from felp.reweighting.outcome import ReweightingOutcome
from felp.reweighting.prep import ReweightingPrep

_ESTIMATOR = (
    "global self-normalized importance sampling over QC-filtered frames "
    "(frame-level SNIS), with microstate summaries derived by binning"
)


class FrameLevelEngine:
    """Single-ensemble SNIS without per-state factor independence."""

    method_id = "frame"

    def aggregate(
        self,
        prep: ReweightingPrep,
        *,
        config: ReweightingConfig,
        sampling_metadata: dict[str, object] | None = None,
    ) -> ReweightingOutcome:
        frame_config = config.frame
        qc_config = prep.qc_config
        if frame_config.qc_policy == "inherit":
            frame_qc_config = qc_config
        else:
            from felp.core import ReweightingQCConfig

            frame_qc_config = ReweightingQCConfig(
                policy=frame_config.qc_policy,
                method=qc_config.method,
                weight_treatment=qc_config.weight_treatment,
                min_frame_count=qc_config.min_frame_count,
                max_frame_weight_fraction=qc_config.max_frame_weight_fraction,
                min_state_ess=qc_config.min_state_ess,
                max_leave_one_out_shift_kj_mol=qc_config.max_leave_one_out_shift_kj_mol,
                max_excluded_frames_per_state=qc_config.max_excluded_frames_per_state,
            )

        excluded_by_state = {
            state_id: set(state.excluded_frame_indices)
            for state_id, state in prep.state_data.items()
        }
        frames = []
        for definition in prep.atlas.definitions:
            for frame in prep.filtered_grouped.get(definition.state_id, []):
                if frame.frame_index in excluded_by_state.get(definition.state_id, set()):
                    continue
                frames.append(frame)

        if not frames:
            frames = [
                frame
                for definition in prep.atlas.definitions
                for frame in prep.filtered_grouped.get(definition.state_id, [])
            ]

        weights = np.asarray(
            [
                math.exp(-prep.beta * (frame.delta_u_kj_mol - prep.global_delta_shift))
                for frame in frames
            ],
            dtype=float,
        )
        qc_result = apply_global_frame_qc(
            frames,
            weights,
            qc_config=frame_qc_config,
            max_excluded_frames=frame_config.max_excluded_frames_global,
        )

        weight_sum = float(qc_result.weights.sum())
        normalized = (
            qc_result.weights / weight_sum if weight_sum > 0.0 else qc_result.weights
        )

        ref_counts: dict[str, float] = defaultdict(float)
        target_weights: dict[str, float] = defaultdict(float)
        for frame, weight in zip(qc_result.frames, normalized, strict=True):
            ref_counts[frame.microstate_id] += 1.0
            target_weights[frame.microstate_id] += float(weight)

        total_ref = sum(ref_counts.values())
        reweight_factors: dict[str, float] = {}
        for definition in prep.atlas.definitions:
            state_id = definition.state_id
            ref_prob = prep.state_data[state_id].reference_probability
            target_mass = target_weights[state_id]
            if ref_prob > 0.0:
                reweight_factors[state_id] = target_mass / ref_prob
            else:
                reweight_factors[state_id] = 0.0

        frame_records = [
            {
                "frame_index": frame.frame_index,
                "microstate_id": frame.microstate_id,
                "delta_u_kj_mol": frame.delta_u_kj_mol,
                "phi_degrees": frame.phi_degrees,
                "psi_degrees": frame.psi_degrees,
                "weight": float(weight),
            }
            for frame, weight in zip(qc_result.frames, normalized, strict=True)
        ]

        method_metadata = {
            "frame_reweighting": {
                "global_effective_sample_size": qc_result.global_ess,
                "global_effective_sample_fraction": qc_result.global_ess_fraction,
                "max_normalized_weight": qc_result.max_normalized_weight,
                "excluded_frame_indices": list(qc_result.excluded_frame_indices),
                "frame_qc_status": qc_result.status,
                "frame_count": len(qc_result.frames),
            }
        }

        outcome = finalize_microstate_outcome(
            prep,
            method_id=self.method_id,
            state_reweighting_estimator=_ESTIMATOR,
            reweight_factors=reweight_factors,
            sampling_metadata=sampling_metadata,
            method_metadata=method_metadata,
            overlap_definition="binned_frame_snis_microstate",
        )
        outcome.shared_metadata["global_effective_sample_size"] = qc_result.global_ess
        outcome.shared_metadata["global_effective_sample_fraction"] = (
            qc_result.global_ess_fraction
        )
        outcome.auxiliary = {
            "frame_weights": frame_records,
            "frame_reweighting_summary": method_metadata["frame_reweighting"],
        }
        return outcome
