"""Reweighting outcome types shared by estimator engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from felp.core import MicrostateResult


@dataclass
class ReweightingOutcome:
    """Aggregated outputs from one reweighting backend."""

    method_id: str
    microstate_results: list[MicrostateResult]
    shared_metadata: dict[str, Any] = field(default_factory=dict)
    auxiliary: dict[str, Any] = field(default_factory=dict)

    @property
    def state_reweighting_estimator(self) -> str:
        return str(self.shared_metadata.get("state_reweighting_estimator", ""))
