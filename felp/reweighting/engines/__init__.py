"""Reweighting estimator engines."""

from __future__ import annotations

from felp.core import ConfigurationError
from felp.reweighting.engines.independent import IndependentZwanzigEngine
from felp.reweighting.engines.neighbor_pool import NeighborPoolEngine

VALID_REWEIGHTING_METHODS = frozenset(
    {"independent", "neighbor_pool"}
)

_ENGINE_TYPES = {
    "independent": IndependentZwanzigEngine,
    "neighbor_pool": NeighborPoolEngine,
}


def get_reweighting_engine(method_id: str):
    """Return a reweighting engine instance for the requested method id."""

    normalized = str(method_id).strip()
    if normalized not in VALID_REWEIGHTING_METHODS:
        raise ConfigurationError(
            "reweighting method must be one of: "
            + ", ".join(sorted(VALID_REWEIGHTING_METHODS))
        )
    return _ENGINE_TYPES[normalized]()
