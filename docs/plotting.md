# Plotting FELP results

Plot from the summary produced by the intended processing protocol. The
`microstate_summary.csv` written by the initial scoring run may differ from
the filtered, PSIS-treated and pooled summary written later by `summarize`.

## Landscape and diagnostic plots

```bash
python -m felp felp-plot \
  --microstate-summary outputs/gly_to_ala/summary/microstate_summary.csv \
  --raw-frame-results outputs/gly_to_ala/frame_results.csv \
  --temperature-kelvin 310 \
  --plot-energy-unit kcal_mol --fel-max-energy-kcal-mol 5 \
  --output-dir outputs/gly_to_ala/plots
```

This writes population comparisons, free-energy landscapes, ESS plots and,
when frame scores are supplied, an energy-difference histogram. The available
projections depend on the CV columns in the inputs. `plot_summary.json` records
the files, estimator and display settings. Use `--overwrite` to regenerate an
existing plot directory.

CSV energies remain in kJ/mol. `--plot-energy-unit` changes the display, while
`--temperature-kelvin` determines the probability-to-free-energy conversion.
Use the same temperature as the reweighting calculation. Set common energy
limits when comparing panels.

## State-grid and frame-projected landscapes

State-grid figures plot the populations in `microstate_summary.csv` on the
assignment atlas. Their filenames include `state_probability_landscape` or
`state_free_energy_landscape`.

Frame-projected figures use the measured CV coordinates of individual frames
and their projection weights. Keep `frame_weights.csv` beside the corresponding
summary; the plotter reads it to preserve the processed weights and state
masses. If it is absent, the plotter reconstructs weights from the frame scores
and recorded state factors. That fallback does not contain the same information
as the exported weights, especially after filtering or pooling. Check the
weight-source fields in `plot_summary.json`.

The distinction matters when the plotting grid differs from the assignment
grid. `--fel-bins` changes the resolution of the frame projection; it does not
redefine the states used for reweighting. Supply `--raw-frame-results` for
these projections and the score histogram.

## Protein CVs

Select CV labels that are present in the saved results. For a two-distance
chignolin atlas:

```bash
python -m felp felp-plot \
  --microstate-summary outputs/chignolin/summary/microstate_summary.csv \
  --raw-frame-results outputs/chignolin/frame_results.csv \
  --plot-cv-x ends --plot-cv-y turn --temperature-kelvin 310 \
  --output-dir outputs/chignolin/plots
```

For a one-dimensional T4L atlas, set `--plot-cv-x psi_114` and omit
`--plot-cv-y`. Use the labels from the actual atlas if they differ from these
examples.

## Display and density estimation

The default histogram renderer shows one color per plotting bin. The main
optional controls are:

| Option | Effect |
|---|---|
| `--fel-bins 72` | Number of projection bins per axis |
| `--fel-max-points 200000` | Frame limit for projected landscapes |
| `--fel-plot-style contour` | Filled contours on the estimated grid |
| `--fel-cmap NAME` | Matplotlib colormap for free-energy plots |
| `--probability-cmap NAME` | Separate colormap for probability plots |
| `--fel-estimator gaussian-kernel` | Kernel density estimate for 2D frame projections |
| `--fel-bandwidth-rule state-adaptive` | Per-state bandwidth selection for the kernel estimate |
| `--fel-surface-smooth-sigma-bins VALUE` | Display-only smoothing of the projected probability grid |

Kernel bandwidth and display smoothing affect the appearance of a surface.
They do not change the saved state populations or repair missing reference
sampling. Use the state-grid figures to inspect the actual discrete estimate.
Keep rendering choices consistent across a comparison and retain
`plot_summary.json` with the figures.

## Compare with independent MD

For a dipeptide target trajectory:

```bash
python -m felp md-kl-plot \
  --felp-output-dir outputs/gly_to_ala/summary \
  --raw-frame-results outputs/gly_to_ala/frame_results.csv \
  --topology-file alanine.parm7 --trajectory alanine.nc \
  --phi-psi-residue-number 2 --temperature-kelvin 310 \
  --output-dir outputs/gly_to_ala/md_comparison
```

Supply the topology for the MD trajectory being compared, which may differ
from the FELP reference topology. Protein distance projections can be specified
with `--distance-cv` and the corresponding plotting labels. The target MD is
used here to evaluate the prediction; it is not an input to U or H.

KL plots can be displayed on an energy scale with `--kl-scale-by-kbt`.
`--kl-display-threshold` suppresses small displayed cell contributions; it is
a display control, not a score-filtering threshold. See each command's `--help`
for the available axis and color-scale controls.

## MSM and composite figures

`msm-plot` produces occupancy and embedding plots from an MSM bundle. The
[starter script](../examples/msm_plot_starter.sh) supplies a basic command.
`composite-figure` assembles multi-row Ramachandran panels using a JSON/YAML
configuration; its panel types and settings are defined in
[composite.py](../felp/plotting/composite.py).

The [benchmark runner](reproduce_matrix.md) also generates comparisons of the
processing stages and pooling treatments. Those figures show sensitivity to
processing choices; the manuscript's final page layouts are separate from
these plotting commands.
