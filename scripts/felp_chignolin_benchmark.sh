#!/bin/bash

# Conda environment
source /sb/bpbrownlab/shared/apps/miniconda3/etc/profile.d/conda.sh 
conda activate compchem

# Convenience
FELP=/net/vivaldi/vivaldi/ssd1/brownbp1/workspace/bpBrownLab/FELP/felp.py
REFERENCE_DIR=/home/brownbp1/Projects/FELP/chignolin_wild-type
PREFIX=`echo $REFERENCE_DIR | awk -F/ '{print $NF}'`

python $FELP run \
--topology-file ${REFERENCE_DIR}/stripped.${PREFIX}.parm7 \
--trajectory ${REFERENCE_DIR}/${PREFIX}_prod.0{001..250}.nc \
--target-residue-number 4 --target-residue-name ALA --original-residue-name PRO \
--output-dir ${REFERENCE_DIR}/test_v2 \
--prior-probability-mode raw_counts \
--mute-pyrosetta --cpu \
--distance-cv ends:1:CA:10:CA \
--distance-cv turn:3:CA:6:CA \
--energy-backend openmm \
--restraint-force-constant 10000 \
--openmm-restraint-force-constant 1000000 \
--rosetta-distance-constraint-sd-angstrom 0.001 \
--minimization-max-iterations 0 \
--temperature-kelvin 310 \
--write-pose-pdbs \
--mobility-mode sequence_window \
--mobility-sequence-window 1 \
--mobility-include-backbone \
--rosetta-backbone-restraint-residue 3 \
--rosetta-backbone-restraint-residue 4 \
--rosetta-backbone-restraint-residue 5 \
--rosetta-target-backbone-restraint-kind phi \
--rosetta-target-backbone-restraint-kind psi \
--rosetta-target-backbone-restraint-kind omega \
--rosetta-target-backbone-restraint-force-constant 25000 \
--rosetta-target-backbone-restraint-tolerance-degrees 15.0 \
--workers 128 --chunk-size 100 \
--overwrite

python $FELP felp-plot \
--microstate-summary /home/brownbp1/Projects/FELP/chignolin_wild-type/test_v2_qc_auto_exclude/microstate_summary.csv \
--raw-frame-results /home/brownbp1/Projects/FELP/chignolin_wild-type/test_v2/frame_results.csv \
--output-dir /home/brownbp1/Projects/FELP/chignolin_wild-type/test_v2_qc_auto_exclude/plots_gaussian_kernel \
--temperature-kelvin 310 \
--plot-cv-x ends \
--plot-cv-y turn \
--fel-max-energy-kj-mol 16 \
--fel-bins 200 \
--fel-estimator gaussian-kernel \
--fel-bandwidth auto \
--fel-bandwidth-rule scott \
--plot-energy-unit kcal_mol --fel-cmap jet \
--overwrite \
