# Reweighting

`manuscript.py` selects the paper's `jctc2026` settings. `orchestrator.py`
prepares the inputs once and dispatches the requested estimator.

- `prep.py`: preserve reference counts, filter scores and calculate local factors.
- `global_gpd.py`: find the first passing global tail cut.
- `psis.py`: apply ArviZ PSIS while preserving relative state-factor scale.
- `topology.py`: construct the neighbor graph, including periodic axes.
- `engines/neighbor_pool.py`: ESS-weighted blending and L-curve selection.
- `engines/_finalize.py`: normalize populations and report U/H diagnostics.

State definitions and frame assignments are handled by `atlas.py`,
`assignment.py` and the grid/coordinate helpers. See the
[theory](../../docs/theory.md) for the estimator derivation,
[neighbor pooling](../../docs/neighbor_pool_reweighting.md) for the illustrated
stencils and L-curve equations, and
[harmonic exposure](../../docs/harmonic_exposure.md) for the Dirichlet problem.
