# Source layout

`__main__.py` dispatches commands. `app.py` connects configuration, frame scoring,
state assignment and reweighting.

| Package | Responsibility |
|---|---|
| `cli`, `core` | Arguments, configuration and result types |
| `io`, `parallel` | Trajectory reading, workers and output |
| `conversion`, `perturbation` | Rosetta conversion, mutation and relaxation |
| `restraints`, `energy` | CV restraint construction and energy evaluation |
| `reweighting` | State assignments, score filtering, PSIS and population estimates |
| `diagnostics` | Weight diagnostics and harmonic exposure |
| `plotting`, `msm` | Landscape plots and MSM workflows |
| `sidechain_ensemble` | Local side-chain sampling for the EGFR sensitivity study |

For the paper's calculation, start with `reweighting/manuscript.py`, then
`reweighting/prep.py`, `reweighting/engines/neighbor_pool.py` and
`reweighting/engines/_finalize.py`. Harmonic exposure is implemented in
`diagnostics/harmonic.py`.

The [theory guide](../docs/theory.md) explains the estimator and its assumptions.
The [manuscript protocol](../docs/manuscript_protocol.md) connects the published
settings to the CLI; [usage](../docs/usage.md) describes the input and output files.
