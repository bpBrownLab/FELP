"""Independent per-state Zwanzig reweighting."""

from __future__ import annotations

from typing import TYPE_CHECKING

from felp.core import ReweightingConfig
from felp.reweighting.engines._finalize import finalize_microstate_outcome
from felp.reweighting.outcome import ReweightingOutcome
from felp.reweighting.prep import ReweightingPrep

if TYPE_CHECKING:
    pass

_ESTIMATOR = (
    "pi_target(i) proportional to pi_reference(i) * <exp(-beta * delta_u)>_i"
)


class IndependentZwanzigEngine:
    """Independent state-local Zwanzig estimator."""

    method_id = "independent"

    def aggregate(
        self,
        prep: ReweightingPrep,
        *,
        config: ReweightingConfig,
        sampling_metadata: dict[str, object] | None = None,
    ) -> ReweightingOutcome:
        del config
        reweight_factors = {
            state_id: state.local_factor
            for state_id, state in prep.state_data.items()
        }
        return finalize_microstate_outcome(
            prep,
            method_id=self.method_id,
            state_reweighting_estimator=_ESTIMATOR,
            reweight_factors=reweight_factors,
            sampling_metadata=sampling_metadata,
        )
