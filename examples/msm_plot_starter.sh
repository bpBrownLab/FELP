#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="outputs/msm_bundle"
OUTPUT_DIR="outputs/msm_publication_plots"

python scripts/msm_plot.py \
  --msm-bundle-dir "${BUNDLE_DIR}" \
  --state-level cluster \
  --output-dir "${OUTPUT_DIR}" \
  --temperature-kelvin 300 \
  --bins 120 \
  --max-state-bars 40
