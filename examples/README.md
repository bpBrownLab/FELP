# Examples

Run these from the repository root. Edit the input paths and system settings in
the shell scripts before running them.

| File | Purpose |
|---|---|
| `dipeptide_microstates.json` | Small example of file-defined phi/psi states |
| `msm_features_dipeptide.json` | Features for fitting a dipeptide MSM |
| `msm_fit_starter.sh` | Fit an MSM and save its state assignments |
| `msm_plot_starter.sh` | Plot an MSM bundle |
| `felp_run_from_bundle_starter.sh` | Score a mutation using an existing bundle |
| `felp_plot_starter.sh` | Plot saved FELP results |
| `egfr_l858r_basins.json` | Basin definitions for EGFR analysis |
| `harmonic_exposure.json` | Five-cell example with H=0.2 and U=0.64 |

The small example state definitions are not the full paper atlases. See the
[quick start](../README.md#score-summarize-and-plot) for the paper's dipeptide grid
and the [harmonic-exposure guide](../docs/harmonic_exposure.md#worked-example)
for the calculation behind the five-cell example. The
[plotting guide](../docs/plotting.md) explains how to use the saved summaries
and frame weights to generate landscape figures.
