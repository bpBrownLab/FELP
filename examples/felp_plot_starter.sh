#!/usr/bin/env bash
set -euo pipefail

MICROSTATE_SUMMARY="outputs/gly_to_ala_from_bundle/microstate_summary.csv"
FRAME_RESULTS="outputs/gly_to_ala_from_bundle/frame_results.csv"
OUTPUT_DIR="outputs/gly_to_ala_publication_plots"

python scripts/felp_plot.py \
  --microstate-summary "${MICROSTATE_SUMMARY}" \
  --raw-frame-results "${FRAME_RESULTS}" \
  --output-dir "${OUTPUT_DIR}" \
  --temperature-kelvin 300 \
  --top-states 40
