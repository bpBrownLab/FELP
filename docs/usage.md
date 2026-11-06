# Using FELP

## Inputs and state assignments

Supply a topology and one or more trajectories. Repeat `--trajectory` for
independent simulations. Files within one flag are consecutive segments:

```bash
--trajectory replica1.001.nc replica1.002.nc \
--trajectory replica2.001.nc replica2.002.nc
```

State definitions can come from a phi/psi grid, a JSON/YAML file or an MSM bundle:

| Source | Options |
|---|---|
| Backbone grid | `--microstate-source grid --phi-bins 45 --psi-bins 45` |
| State file | `--microstate-source file --microstate-file atlas.json` |
| MSM bundle | `--microstate-source msm_bundle --msm-bundle-dir reference.msm_bundle` |

Use `--prior-probability-mode raw_counts` for the empirical reference populations.
Use `loaded` when the state file or bundle supplies the desired prior.
`examples/dipeptide_microstates.json` illustrates the state-file format.

The example defines a few basins rather than a complete grid. To run it while
excluding frames outside those basins, use `--allow-unassigned`; the resulting
populations are conditional on the assigned region. The manuscript protocol
requires a complete uniform atlas, including empty cells. See
[theory](theory.md#reference-populations-and-state-definitions) for how priors
and coverage counts enter the calculation.

## Restraints and mobility

Residue selectors use one-based pose indices. Check these against the prepared
protein: they may differ from PDB numbering. CV targets are measured on each
reference frame and applied to both scoring branches.

For a custom four-atom torsion, specify its label and four residue/atom pairs.
For example, the EGFR activation-loop coordinate in the supplied numbering is:

```bash
--dihedral-cv aloop:159:CA:160:CA:161:CA:162:CA
```

Use `--distance-cv` for atom-pair distances and `--cv` for backbone torsions;
`python -m felp run --help` lists their syntax. Set the microstate atlas to the
same CVs used for assignment.

For example, the chignolin distance coordinates are specified as
`--distance-cv ends:1:CA:10:CA --distance-cv turn:3:CA:6:CA`, and the T4L
backbone coordinate as `--cv psi:114`. Custom labels such as `ends`, `turn` and
`aloop` must match those in the atlas. The
[manuscript protocol](manuscript_protocol.md#scoring-settings) lists the
restraints and mobility settings used for the benchmark systems.

`--restraint-force-constant` sets the shared restraint strength.
`--rosetta-restraint-force-constant` and `--openmm-restraint-force-constant`
override it for each stage. Rosetta distance restraints use a width in Angstrom,
set by `--rosetta-distance-constraint-sd-angstrom`.

The default mobility is the mutation site's side chain. A sequence window or
spatial radius can permit nearby residues to move; backbone motion is a separate
option. These choices affect the energies, so keep them fixed when comparing
runs. `--cpu` and `--gpu` select the OpenMM platform.

## Reprocess saved scores

`run` writes the per-frame energies. `summarize` reads those energies without
repeating molecular scoring. For a protein atlas:

```bash
python -m felp summarize \
  --protocol jctc2026 --raw-frame-results outputs/mutant/frame_results.csv \
  --microstate-source file --microstate-file atlas.json \
  --prior-probability-mode raw_counts --temperature-kelvin 310 \
  --output-dir outputs/mutant/summary
```

Add `--neighbor-pool-periodic-cvs aloop` for a periodic axis named `aloop`.
Names must match the atlas. Distance axes are nonperiodic.
Use `--repartition` only when assigning saved CV measurements to a new atlas.
For subsampled scores, retain the original assignments and full reference counts.
Use the reference simulation temperature.

Add `--resume` to a scoring run to continue from saved results. To reduce scoring
cost, `--microstate-sample-fraction 0.1 --microstate-min-samples 10` selects a
subset after an assignment pass.

Keep `sampling_manifest.json` alongside a subsampled `frame_results.csv`.
`summarize` reads it from the same directory to recover the full reference
counts. Repartitioning subsampled scores alone cannot reconstruct the full
reference counts on a new atlas; use the original assignments and atlas.

To combine separate scoring runs before summarizing:

```bash
python scripts/merge_frame_results.py outputs/part1 outputs/part2 \
  --output outputs/merged/frame_results.csv
```

Use compatible columns, units and scoring settings. The merger assigns unique
frame and trajectory IDs and keeps the original trajectory IDs in metadata.

## Outputs

| File | Contents |
|---|---|
| `run_config.json` | Resolved scoring configuration |
| `frame_results.csv` | Per-frame energies, scores, assignments and metadata |
| `sampling_manifest.json` | Reference counts and selected frames from the sampling pass |
| `microstate_summary.csv` | Reference and predicted populations, factors and ESS |
| `frame_weights.csv` | Exported frame weights for projections of the processed result |
| `reweighted_surface.csv` | Landscape coordinates and free energies |
| `run_summary.json` | Processing settings, exclusions and diagnostics |

Multiple estimators write separate `reweighting_METHOD/` subdirectories.
Under `jctc2026`, U and H are recorded in
`run_summary.json` at `overlap_diagnostics.manuscript_diagnostics`.

Keep the summary and its frame-weight table together when transferring or
plotting results. See [plotting](plotting.md) for landscape and MD comparisons,
and [diagnostics](overlap_diagnostics.md) for interpreting the reported values.

For implementation details, see the [source layout](../felp/README.md).
