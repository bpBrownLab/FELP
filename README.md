# FELP

Free Energy Landscape Perturbation: restrained, state-local reweighting of a reference MD ensemble under a local perturbation (e.g. point mutation).

Given a reference trajectory and discrete microstates (φ/ψ grid, JSON definitions, or MSM bundle), FELP evaluates per-frame energy differences and aggregates them with a state-local Zwanzig estimator to obtain target-state populations and relative free energies.

## Installation

FELP runs from a git checkout. There is no PyPI package, and `pip install` does not
provide the customized ParmEd that FELP depends on.

```bash
git clone <repository-url> FELP
cd FELP
git submodule update --init ParmEd      # required: bpBrownLab/ParmEd fork with the Rosetta bridge
```

FELP loads the vendored ParmEd from the `ParmEd/` submodule at runtime, so the submodule
must be initialized even if you also `pip install -e .`. Do not rely on a separately
installed `parmed`; the upstream PyPI package is missing the Rosetta `save_rosetta`/
virtual-atom handling FELP requires.

Run from the repository root:

```bash
python felp.py --help
python -m felp --help
```

### Dependencies by command

| Need | Required for |
|------|--------------|
| numpy, scipy, matplotlib, mdtraj, pyyaml (in `requirements.txt`) | All commands |
| pyemma | `msm-fit`, `msm-inspect`, `msm-plot`, `msm-export-samples` |
| ParmEd submodule + PyRosetta + OpenMM | `run` (per-frame mutation and energy evaluation) |

PyRosetta and OpenMM are not pip-installable from `requirements.txt`; install them separately
(PyRosetta requires a license; OpenMM via conda). Plotting and reweighting from existing
`frame_results.csv` (`summarize`, `felp-plot`, `md-kl-plot`, `ess-compare`) do not need
PyRosetta or OpenMM. Method details: [docs/theory.md](docs/theory.md).

## Quick start

The repo ships a small GLY example (`data/GLY/`) for a smoke test. A full `run` performs
the mutation and energy evaluation, so it needs the ParmEd submodule, PyRosetta, and OpenMM:

```bash
python felp.py run \
  --topology-file data/GLY/GLY.dry.parm7 \
  --trajectory data/GLY/GLY_prod.0001.nc \
  --target-residue-number 2 \
  --target-residue-name ALA \
  --output-dir outputs/gly_to_ala \
  --phi-bins 12 --psi-bins 12 \
  --prior-probability-mode raw_counts \
  --workers 4
```

Then plot the results (no PyRosetta/OpenMM needed):

```bash
python felp.py felp-plot \
  --microstate-summary outputs/gly_to_ala/microstate_summary.csv \
  --raw-frame-results outputs/gly_to_ala/frame_results.csv \
  --output-dir outputs/gly_to_ala/plots \
  --temperature-kelvin 300
```

The bundled `data/GLY/` files are a single short trajectory chunk for testing the workflow.
Reproducing the manuscript benchmarks requires your own MD trajectories; those large datasets
are not distributed with the repository.

## Examples

### Trajectory groups

Repeat `--trajectory` for independent simulations; files within one flag are concatenated:

```bash
--trajectory simA.001.nc simA.002.nc \
--trajectory simB.001.nc simB.002.nc
```

### File-defined microstates

```bash
python felp.py run \
  --topology-file data/reference.pdb \
  --trajectory data/reference.dcd \
  --target-residue-number 2 --target-residue-name ALA \
  --microstate-source file \
  --microstate-file examples/dipeptide_microstates.json \
  --prior-probability-mode loaded \
  --output-dir outputs/gly_to_ala_explicit
```

### MSM bundle workflow

Fit a bundle, then run FELP from it:

```bash
python felp.py msm-fit \
  --topology-file data/reference.pdb \
  --trajectory data/reference.001.dcd data/reference.002.dcd \
  --feature-spec-file examples/msm_features_dipeptide.json \
  --output-dir outputs/dipeptide_bundle.msm_bundle \
  --k-clusters 50 --msm-lag 5 --write-plots

python felp.py run \
  --topology-file data/reference.pdb \
  --trajectory data/reference.001.dcd data/reference.002.dcd \
  --target-residue-number 2 --target-residue-name ALA \
  --microstate-source msm_bundle \
  --msm-bundle-dir outputs/dipeptide_bundle.msm_bundle \
  --prior-probability-mode loaded \
  --output-dir outputs/gly_to_ala_from_bundle
```

Legacy entry point (same as `msm-fit`):

```bash
python build_msm.py --help
```

### Resummarize with coupled reweighting

```bash
python felp.py summarize \
  --raw-frame-results outputs/gly_to_ala/frame_results.csv \
  --microstate-source file \
  --microstate-file examples/dipeptide_microstates.json \
  --reweighting-methods independent neighbor_pool \
  --neighbor-pool-solver blend --neighbor-pool-tau-mode lcurve \
  --output-dir outputs/gly_to_ala_resummarized
```

### Frame subsampling

Evaluate a fraction of frames per microstate (cheap assignment pass first):

```bash
python felp.py run ... \
  --microstate-sample-fraction 0.1 \
  --microstate-min-samples 10 \
  --microstate-sampling-seed 0
```

### Other commands

| Command | Purpose |
|---------|---------|
| `validate-microstates` | Check microstate JSON/YAML |
| `summarize` | Recompute summaries from `frame_results.csv` |
| `msm-inspect` | Inspect an exported bundle |
| `msm-plot` | Publication plots from a bundle |
| `msm-export-samples` | Representative structures per state |
| `md-kl-plot` | Compare FELP surface to reference MD |
| `ess-compare` | Bootstrap ESS comparison across methods |

Shorter wrappers: `scripts/felp_run.py`, `scripts/felp_plot.py`, `scripts/msm_fit.py`, `scripts/msm_plot.py`. Shell templates in `examples/`.

Use `--cpu` / `--gpu` for OpenMM platform selection. See [docs/theory.md](docs/theory.md#openmm-platform-notes) for CPU threading defaults.

## Documentation

| Topic | File |
|-------|------|
| Method, diagnostics, and backend overview | [docs/theory.md](docs/theory.md) |
| Neighbor-pool τ / blend | [docs/neighbor_pool_reweighting.md](docs/neighbor_pool_reweighting.md) |
| Legacy entry points | [archive/legacy/README.md](archive/legacy/README.md) |

## Outputs

**`run`** writes `run_config.json`, `frame_results.csv`, `microstate_summary.csv`, `reweighted_surface.csv`, `run_summary.json`, and optionally `sampling_manifest.json`.

**`summarize`** recomputes `microstate_summary.csv` and related files from an existing `frame_results.csv`. Multi-method runs add `reweighting_<method>/` directories.

**`msm-fit`** writes `manifest.json`, `assignments.npz`, `embedding.npz`, and optional validation plots.

**Plotting commands** write `*.png` and `plot_summary.json`. Add `--resume` to continue interrupted runs.

## Testing

```bash
python -m pytest -q
```

The GLY→ALA end-to-end benchmark runs when PyRosetta and OpenMM are available.
