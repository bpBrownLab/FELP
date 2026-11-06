"""Processing settings for the 2026 JCTC paper."""
from __future__ import annotations

from felp.core import ConfigurationError, NeighborPoolConfig, ReweightingConfig, ReweightingQCConfig

PROTOCOL_ID = "jctc2026"


def manuscript_configs(atlas, *, periodic_cv_names=()):
    """Return the published configs, without choosing the atlas or its period.

    Geometry, temperature and source scores are inputs, not fit parameters.
    A complete uniform atlas is required to make the 1D/2D selection explicit.
    """
    from felp.diagnostics.harmonic import grid_from_atlas

    if atlas.reference_probability_mode != "raw_counts":
        raise ConfigurationError("The jctc2026 protocol requires raw_counts reference priors")
    try:
        grid = grid_from_atlas(atlas)
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc
    if set(periodic_cv_names) - set(grid.cv_names):
        raise ConfigurationError("Periodic CV names must be present in the atlas")
    reweighting = ReweightingConfig(
        methods=("neighbor_pool",),
        neighbor_pool=NeighborPoolConfig(
            solver="blend", tau_mode="lcurve", tau=None, min_local_weight=0.,
            grid_connectivity="chebyshev" if len(grid.shape) == 2 else "cardinal",
            chebyshev_radius=3, periodic_cv_names=tuple(periodic_cv_names),
        ),
    )
    qc = ReweightingQCConfig(
        policy="exclude-frames", method="gpd", weight_treatment="gpd-psis",
        exclusion_scope="global", exclusion_strategy="global-gpd-k",
        pareto_k_threshold=.5, min_frame_count=20,
    )
    return reweighting, qc
