# Local side-chain probe

This package implements the EGFR sensitivity calculation in the Supporting
Information. Each reference/mutant branch receives independent jump-rotamer
proposals at fixed backbone coordinates. OpenMM minimizes each proposal, and
`aggregation.py` combines its energy and recorded proposal density into a local
configurational sum.

Add these options to an EGFR scoring command for the selected probe frames:

```bash
--sidechain-ensemble-mode proposal-quadrature \
--sidechain-ensemble-retained-samples 32 --sidechain-ensemble-chains 2
```

This gives 64 proposals per branch. Keep the same restraints, mobility and
force-field settings as the paired single-endpoint calculation. The frame table
retains both the endpoint score and local free-energy estimate; member energies
and proposal densities are written separately. Production uses the default
`--sidechain-ensemble-mode off`.
