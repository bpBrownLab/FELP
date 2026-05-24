#!/bin/bash
# Build the GLY rotation composite figure (Reference | FELP | MD | KL).
#
# Edit data/composite_rotation_mutations.json for fonts, columns, and paths.
# Add optional columns by inserting entries such as:
#   {"type": "delta_probability", "header": "Delta probability"}
#   {"type": "delta_free_energy", "header": "Delta free energy"}
#
# Run from the data tree:
#   bash run_composite_rotation_figure.sh

set -euo pipefail

DATA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FELP="${FELP:-/net/vivaldi/vivaldi/ssd1/brownbp1/workspace/bpBrownLab/FELP/felp.py}"
CONFIG="${CONFIG:-${DATA_ROOT}/composite_rotation_mutations.json}"

python "$FELP" composite-figure \
  --config "$CONFIG" \
  --overwrite
