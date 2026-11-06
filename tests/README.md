# Tests

From the repository root:

```bash
python -m pytest -q -m 'not integration'
```

Install the postprocessing dependencies and pytest first. OpenMM is also needed
for energy tests. Integration tests additionally
need PyRosetta and MDTraj. Run them with `python -m pytest -q -m integration`.

`test_manuscript_protocol.py` checks the processing preset.
`test_manuscript_harmonic_replay.py` reproduces H for all 60 validation cases from
`data/manuscript_harmonic_cases.json.gz`. That fixture contains original state
counts, predicted populations and expected H values; it contains no trajectories.
`test_restraints.py` checks custom-dihedral construction and both scoring backends.

The root `conftest.py` loads the bundled ParmEd so tests use the Rosetta bridge
shipped with FELP.
