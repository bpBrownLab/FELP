#!/usr/bin/env bash
# Run a minimal FELP job: one frame, one worker. Pass topology and trajectory,
# then required felp.py run arguments (target residue, output dir, etc.).
#
# Example:
#   ./scripts/smoke_trajectory_run.sh ref.pdb sim.nc \
#     --target-residue-number 2 --target-residue-name ALA \
#     --original-residue-name GLY --output-dir /tmp/felp_smoke
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TOPO="${1:?usage: $0 <topology_file> <trajectory_file> [extra felp.py run args...]}"
TRAJ="${2:?}"
shift 2
exec python felp.py run \
  --topology-file "$TOPO" \
  --trajectory "$TRAJ" \
  --frame-stop 1 \
  --workers 1 \
  "$@"
