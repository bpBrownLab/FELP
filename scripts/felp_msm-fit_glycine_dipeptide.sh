#!/bin/bash

python /vivaldi/ssd1/brownbp1/workspace/bpBrownLab/FELP/felp.py msm-fit \
	--topology-file /home/murdocte/Projects/Rotation/GLY/GLY.dry.parm7 \
	--trajectory /home/murdocte/Projects/Rotation/GLY/GLY_prod.nc \
	--output-dir /home/brownbp1/Projects/FELP/msm_bundle_GLY
	--phi-psi-residue-number 2
	--reducer none
	--msm-lag 2 \
	--metastable-states 6 \
	--its-lags 1 2 3 4 5 6 7 8 9 10 \
	--its-processes 3  \
	--overwrite \
	--clustering-method kmeans \
	--k-clusters 100  \
	--k-iter 100 \
	--ck-mlags 100  \
	--overwrite \
	--write-plots \
	--save-pyemma-checkpoints \
	--validation-cluster-scatter-max-points 0
