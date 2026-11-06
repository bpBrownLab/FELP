"""Numerically stable local-ensemble aggregation.

These estimators operate on OpenMM-minimized member energies. They are
configurational-integral approximations over a local rotamer subspace, not
absolute mutation free energies. Sequence-wide additive constants cancel only
after the per-state factors are normalized into a conformational landscape.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Iterable, Sequence

import numpy as np

from felp.core import ConfigurationError, ReweightingError


def stable_logsumexp(values: Sequence[float]) -> float:
    """Return log-sum-exp of ``values`` with a finite-max shift."""

    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ReweightingError("log-sum-exp requires at least one finite value.")
    if not np.isfinite(array).all():
        raise ReweightingError("log-sum-exp requires finite values.")
    shift = float(array.max())
    return shift + math.log(float(np.exp(array - shift).sum()))


def require_proposal_density(density: float | None, *, index: int = 0) -> float:
    """Require a finite, positive density for each proposal."""

    if density is None or not math.isfinite(float(density)) or float(density) <= 0.0:
        raise ReweightingError(
            "proposal-quadrature requires a positive recorded PyRosetta proposal "
            f"density for every draw; member {index} is missing or invalid."
        )
    return float(density)


def proposal_corrected_log_weights(
    energies_kj_mol: Sequence[float],
    proposal_densities: Sequence[float | None],
    beta: float,
) -> np.ndarray:
    """Return ``-beta * E - log q`` terms for proposal-corrected quadrature."""

    if len(energies_kj_mol) != len(proposal_densities):
        raise ReweightingError(
            "proposal-quadrature energies and proposal densities must align."
        )
    if not energies_kj_mol:
        raise ReweightingError("proposal-quadrature requires at least one member.")
    terms = []
    for index, (energy, density) in enumerate(
        zip(energies_kj_mol, proposal_densities, strict=True)
    ):
        if not math.isfinite(float(energy)):
            raise ReweightingError(
                f"proposal-quadrature member {index} has a non-finite OpenMM energy."
            )
        q = require_proposal_density(density, index=index)
        terms.append(-float(beta) * float(energy) - math.log(q))
    return np.asarray(terms, dtype=float)


def proposal_quadrature_log_z(
    energies_kj_mol: Sequence[float],
    proposal_densities: Sequence[float | None],
    beta: float,
) -> float:
    """Estimate ``log Z`` with independent proposal-corrected log-sum-exp.

    ``log Z ≈ logsumexp(-β E_i - log q_i) - log N``.
    """

    terms = proposal_corrected_log_weights(energies_kj_mol, proposal_densities, beta)
    return stable_logsumexp(terms) - math.log(float(len(terms)))


def local_free_energy_kj_mol(log_z_perturbed: float, log_z_reference: float, beta: float) -> float:
    """Return ``ΔA = -β^{-1} (log Z_1 - log Z_0)`` from local partition estimates."""

    if float(beta) <= 0.0:
        raise ConfigurationError("Local-ensemble aggregation requires a positive beta.")
    return -(float(log_z_perturbed) - float(log_z_reference)) / float(beta)


def importance_ess(log_weights: Sequence[float]) -> float:
    """Return the self-normalized importance-sampling ESS of log-weights."""

    array = np.asarray(list(log_weights), dtype=float)
    if array.size == 0:
        return 0.0
    shift = float(array.max())
    weights = np.exp(array - shift)
    total = float(weights.sum())
    if total <= 0.0:
        return 0.0
    return float((total**2) / max(float(np.square(weights).sum()), 1.0e-12))


def proposal_weight_ess(
    energies_kj_mol: Sequence[float],
    proposal_densities: Sequence[float | None],
    beta: float,
) -> float:
    """ESS of the proposal-corrected quadrature weights."""

    return importance_ess(
        proposal_corrected_log_weights(energies_kj_mol, proposal_densities, beta)
    )


def rotamer_basin_counts(assignments: Iterable[str]) -> dict[str, int]:
    """Count rotamer-basin labels while retaining multiplicity."""

    return dict(Counter(str(item) for item in assignments))


def classify_rotamer_assignment(chi_degrees: Sequence[float]) -> str:
    """Map chi angles to a diagnostic rotamer-basin label.

    Standard three-fold bins are used: plus (60), trans (180), minus (-60).
    Classification is diagnostic only; samples are never collapsed by label.
    """

    labels: list[str] = []
    for chi in chi_degrees:
        angle = ((float(chi) + 180.0) % 360.0) - 180.0
        if -120.0 <= angle < 0.0:
            labels.append("m")
        elif 0.0 <= angle < 120.0:
            labels.append("p")
        else:
            labels.append("t")
    return ":".join(labels) if labels else "none"
