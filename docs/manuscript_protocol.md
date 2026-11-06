# Manuscript protocol

The JCTC calculation combines restrained endpoint scoring, a global score-tail
cut, statewise PSIS and neighbor pooling. `--protocol jctc2026` selects the
postprocessing settings. CVs, temperature, restraints and residue mobility are
inputs to the calculation and must be set for the system being studied.

## Reprocess saved scores

For the dipeptide atlas:

```bash
python -m felp summarize \
  --protocol jctc2026 \
  --raw-frame-results outputs/gly_to_ala/frame_results.csv \
  --microstate-source grid --phi-bins 45 --psi-bins 45 \
  --prior-probability-mode raw_counts \
  --neighbor-pool-periodic-cvs phi psi --temperature-kelvin 310 \
  --output-dir outputs/gly_to_ala/summary
```

For protein CVs, replace the grid options with
`--microstate-source file --microstate-file atlas.json`. The atlas must be
a complete uniform 1D or 2D grid. Use `--repartition` when saved CV measurements
need reassignment to the specified atlas; retain the original assignments when
the atlas is unchanged. The preset does not repeat molecular scoring.

The paper uses 310 K. For other datasets, use the reference simulation
temperature. Specify periodic axes explicitly: `phi psi` for dipeptides,
`psi_114` for T4L and `aloop` for EGFR. Chignolin's distance axes do not wrap.

## Processing settings

The preset overrides individual QC and pooling settings as follows:

| Operation | Setting |
|---|---|
| Reference prior | Raw counts fixed before score-tail deletion |
| Global cut | Stable ascending score deletion; first finite global $k<0.5$ |
| Minimum retained sample | 20 scores over the whole dataset |
| PSIS | ArviZ 0.22.0, within each state, `reff=1` |
| State-factor scale | Preserved across statewise PSIS normalization |
| Pooling | ESS-weighted blend; L-curve $\tau$; no minimum local weight |
| Pooling graph | Chebyshev radius 3 in 2D; cardinal in 1D |
| State-level QC | Diagnostic only; no further per-state score deletion |
| U | Final target probability in states with local treated-weight ESS < 10 |
| H | Square-root probability continuation on the original cardinal grid |

The individual CLI name for ArviZ weight treatment is `gpd-psis`. Generic CLI
defaults do not select the complete manuscript protocol; use the preset or
the [uniform matrix runner](reproduce_matrix.md).

`run_summary.json` records the resolved configuration under
`processing_protocol`, the tail fit and exclusions under `reweighting_qc`,
and U/H under `overlap_diagnostics.manuscript_diagnostics`. The full harmonic
calculation status is under `overlap_diagnostics.harmonic_exposure`.

If no cut passes, the original scores remain and QC is unresolved. Constant
identity weights have an undefined tail fit; this is the GLY→GLY control in
the paper. A passing trimmed fit does not establish the physical correctness
of the scores or coverage of unvisited target conformations.

The cut preserves the reference prior and visited-cell set, but it can remove
every usable score from a state. Such a state has zero local ESS, may borrow
a pooled factor, and remains an H anchor. For subsampled scoring, keep the
original assignment counts and `sampling_manifest.json`; see
[reference populations](theory.md#reference-populations-and-state-definitions).

## Scoring settings

Both branches start from the same reference frame. Production preparation uses
constrained Cartesian Rosetta FastRelax, with five cycles by default, followed
by OpenMM minimization with `amber/protein.ff14SBonlysc.xml` and
`implicit/gbn2.xml` using mbondi3 radii. Base CV restraints are applied to both
branches in both stages. Atoms outside the mobile set are fixed during OpenMM
minimization. Production endpoints do not thermally sample side-chain entropy.

| System | Base CVs and restraints | Mobile residues |
|---|---|---|
| Dipeptides | Residue 2 phi/psi, 45 × 45 grid; force constant 100000 in both stages | Mutation-site side chain |
| Chignolin | CA1–CA10 and CA3–CA6 distances; Rosetta width 0.001 Å; OpenMM force constant 1000000 | Residues 1–2 and 9–10, backbone and side chains |
| T4L | psi114; Rosetta force constant 10000; OpenMM force constant 1000000 | Mutation-site side chain |

Chignolin also uses Rosetta-only phi/psi restraints on residues 1, 2, 9 and 10.
T4L uses Rosetta-only phi/psi restraints at mutation site 99. Both use force
constant 25000 and a 15° validation tolerance for these additional restraints.
These numeric constants belong to the respective restraint implementations;
Rosetta and OpenMM do not share a universal restraint-unit convention.

EGFR uses a custom four-CA activation-loop dihedral. In the supplied pose
numbering, its CV specification is:

```bash
--dihedral-cv aloop:159:CA:160:CA:161:CA:162:CA
```

Use the original scoring configuration when reproducing a system. The
postprocessing preset does not supply these restraint or mobility choices.
See [usage](usage.md#restraints-and-mobility) for the relevant CLI options.

## Sensitivity analyses

The matrix contains 67 systems: 60 dipeptide transformations, two chignolin,
three T4L and two EGFR cases. Each receives six pooling treatments: cardinal
and Chebyshev radius 3, each with $\tau=1$, $\tau=100$ and L-curve $\tau$.
The filtered and PSIS-treated sample is prepared once per system and reused.
See [benchmark reproduction](reproduce_matrix.md) for commands and outputs.

The EGFR local side-chain reassessment is a separate 150-frame sensitivity
study. It uses 64 jump-rotamer proposals per branch in two batches of 32 and
proposal-corrected local partition estimates. Those values do not replace the
production endpoint scores. The
[side-chain module guide](../felp/sidechain_ensemble/README.md) documents it.
