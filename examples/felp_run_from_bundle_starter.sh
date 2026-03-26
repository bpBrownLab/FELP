#!/usr/bin/env bash
set -euo pipefail

TOPOLOGY="path/to/reference.pdb"
BUNDLE_DIR="outputs/msm_bundle"
OUTPUT_DIR="outputs/gly_to_ala_from_bundle"

python scripts/felp_run.py \
  --topology-file "${TOPOLOGY}" \
  --trajectory path/to/simA.001.nc path/to/simA.002.nc path/to/simA.003.nc \
  --trajectory path/to/simB.001.nc path/to/simB.002.nc \
  --target-residue-number 2 \
  --target-residue-name ALA \
  --microstate-source msm_bundle \
  --msm-bundle-dir "${BUNDLE_DIR}" \
  --state-level cluster \
  --output-dir "${OUTPUT_DIR}" \
  --workers 4
