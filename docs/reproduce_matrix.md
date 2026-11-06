# Reproduce the benchmark matrix

`scripts/manuscript_systems.json` lists 67 systems: 60 dipeptide directions,
two chignolin directions, three T4L mutations and two EGFR mutations.
Each has six neighbor treatments: cardinal and Chebyshev-3, with tau=1,
tau=100 and L-curve tau. The filtered/PSIS scores are prepared once per system.

Install `scripts/requirements_postprocess.txt` and supply the original score
and atlas files. Paths in the manifest are relative to `--project-root`.
For a different data layout, copy the manifest and edit its input paths.

Prepare and run one system:

```bash
python scripts/manuscript_uniform_matrix.py \
  --prepare outputs/matrix --project-root /path/to/FELP_data \
  --system GLY_to_ALA --workers 4

python scripts/manuscript_uniform_matrix.py \
  --spec outputs/matrix/control/specs/GLY_to_ALA.json
```

Repeat `--system` for several cases, or omit it to prepare all 67. Use
`--manifest systems.json` for a custom manifest. Preparation writes task specs;
it does not submit cluster jobs. Execute each generated spec to run the matrix.

Final tables are written to:

```text
results/SYSTEM/comparisons/TREATMENT/treated/reweighting_neighbor_pool/microstate_summary.csv
```

Select `cheb3_lcurve` for the 2D paper results and `cardinal_lcurve` for 1D.
Each result includes the resolved settings, input/code hashes, exclusions and
U/H diagnostics. `--resume` reuses completed tasks only when their recorded
inputs and code match. No MD or molecular scoring is run by this script.

Use `python -m felp felp-plot` for landscape plots; see the
[plotting guide](plotting.md) for examples. The matrix's comparison
figures show processing sensitivity; they do not reproduce the paper's page layout.
