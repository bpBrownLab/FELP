#!/bin/bash
# Re-summarize rotation mutation runs onto a uniform Ramachandran grid:
#   45×45 phi/psi bins, raw-count reference priors, --repartition,
#   exclude-frames QC, multi-method reweighting (neighbor_pool L-curve blend).
#
# Applies to GLY, PRO, and ALA (ignores whatever microstate definition the
# original FELP run used — MSM clusters, auto_microstates.json, etc.).
#
# Run from the data tree (e.g. ~/Projects/FELP/data):
#   bash run_reweight_lcurve.sh
#   RESIDUE=GLY PARALLEL_JOBS=8 bash run_reweight_lcurve.sh
#   RESIDUE=ALA TARGET=TRP bash run_reweight_lcurve.sh

set -euo pipefail

DATA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FELP="${FELP:-/net/vivaldi/vivaldi/ssd1/brownbp1/workspace/bpBrownLab/FELP/felp.py}"
PHI_BINS="${PHI_BINS:-45}"
PSI_BINS="${PSI_BINS:-45}"
PARALLEL_JOBS="${PARALLEL_JOBS:-6}"
RESIDUES="${RESIDUES:-GLY PRO ALA}"
TARGET="${TARGET:-}"

if [[ -n "${RESIDUE:-}" ]]; then
  RESIDUES="$RESIDUE"
fi

summarize_one() {
  local indir="$1"
  local outdir="${indir/_full.from_msm_bundle/_reweight_lcurve}"
  local residue

  residue=$(basename "$indir" | sed -n 's/^felp_\([^_]*\)_to_.*/\1/p')
  echo "[$(date -Iseconds)] ${indir##*/} (${residue}, ${PHI_BINS}x${PSI_BINS} grid) -> ${outdir##*/}"

  python "$FELP" summarize \
    --raw-frame-results "${indir}/frame_results.csv" \
    --output-dir "$outdir" \
    --microstate-source grid \
    --phi-bins "$PHI_BINS" \
    --psi-bins "$PSI_BINS" \
    --prior-probability-mode raw_counts \
    --repartition \
    --temperature-kelvin 310 \
    --workers 1 \
    --overwrite \
    --reweighting-qc-policy exclude-frames \
    --reweighting-methods independent neighbor_pool laplacian frame \
    --neighbor-pool-solver blend \
    --neighbor-pool-tau-mode lcurve \
    --laplacian-lambda 1.0
}

export -f summarize_one
export FELP PHI_BINS PSI_BINS

RUN_DIRS=()
for res in $RESIDUES; do
  while IFS= read -r d; do
    RUN_DIRS+=("$d")
  done < <(
    find "${DATA_ROOT}/${res}" -maxdepth 1 -type d -name 'felp_*_full.from_msm_bundle' | sort
  )
done

if [[ -n "$TARGET" ]]; then
  if [[ -n "${RESIDUE:-}" ]]; then
    RUN_DIRS=("${DATA_ROOT}/${RESIDUE}/felp_${RESIDUE}_to_${TARGET}_full.from_msm_bundle")
  else
    echo "Set RESIDUE=GLY|PRO|ALA when using TARGET=" >&2
    exit 1
  fi
fi

if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
  echo "No input dirs found under ${DATA_ROOT} for: ${RESIDUES}" >&2
  exit 1
fi

echo "Summarizing ${#RUN_DIRS[@]} runs (${PHI_BINS}x${PSI_BINS} Rama grid, parallel=${PARALLEL_JOBS})"
printf '%s\n' "${RUN_DIRS[@]}" | xargs -r -P "$PARALLEL_JOBS" -I {} bash -c 'summarize_one "$1"' _ {}
