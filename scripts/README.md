# Utilities

Run scripts from the repository root with the same Python environment as FELP.
Each command accepts `--help`.

| Script | Purpose |
|---|---|
| `felp_run.py`, `felp_plot.py` | Wrappers for the scoring and plotting commands |
| `msm_fit.py`, `msm_plot.py` | Wrappers for MSM fitting and plotting |
| `merge_frame_results.py` | Combine score tables while preserving source identities |
| `manuscript_uniform_matrix.py` | Run the paper's six pooling treatments per system |
| `build_release.py` | Build a source archive from the release manifest |

`manuscript_postprocess.py` and `manuscript_postprocess_analysis.py` provide
planning, analysis and plotting helpers for the matrix runner. Use
`manuscript_uniform_matrix.py` to reproduce the paper's settings.
`_dispatch.py` supports the command wrappers.

The input paths are listed in `manuscript_systems.json`. See
[benchmark reproduction](../docs/reproduce_matrix.md) for commands and outputs.
