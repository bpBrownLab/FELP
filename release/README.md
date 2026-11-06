# Source archive

`jctc2026_manifest.json` lists the files included in the source distribution.
Keep it updated when adding examples, tests or runtime data.

From the repository root:

```bash
python scripts/build_release.py --output-dir dist
```

The builder checks file paths and local imports, then writes a gzip archive,
SHA-256 checksum and file inventory. It includes the modified ParmEd source.
Use a new output directory for each build; existing archives are not overwritten.
