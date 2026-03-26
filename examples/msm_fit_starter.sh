#!/usr/bin/env bash
set -euo pipefail

TOPOLOGY="path/to/reference.pdb"
FEATURE_SPEC="examples/msm_features_dipeptide.json"
OUTPUT_DIR="outputs/msm_bundle"

python scripts/msm_fit.py \
  --topology-file "${TOPOLOGY}" \
  --trajectory path/to/simA.001.nc path/to/simA.002.nc path/to/simA.003.nc \
  --trajectory path/to/simB.001.nc path/to/simB.002.nc \
  --feature-spec-file "${FEATURE_SPEC}" \
  --output-dir "${OUTPUT_DIR}" \
  --stride 1 \
  --dt-traj "10 ps" \
  --reducer tica \
  --reducer-lag 10 \
  --clustering-method kmeans \
  --k-clusters 100 \
  --msm-lag 5 \
  --metastable-states 4 \
  --write-plots
