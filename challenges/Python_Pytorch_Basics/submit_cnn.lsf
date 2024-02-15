#!/bin/bash
#BSUB -P TRN022
#BSUB -W 01:00
#BSUB -nnodes 1
#BSUB -J pytorch_cnn
#BSUB -o pytorch_cnn.%J.out
#BSUB -e pytorch_cnn.%J.err
#BSUB -alloc_flags smt4 gpumps

cd $LSB_OUTDIR
date

module load python
module load cuda

source activate /gpfs/wolf/world-shared/stf007/9b8/public_envs/opence_clone
export MPLCONFIGDIR="${MEMBERWORK}/trn022/mpl_cache"

jsrun -n1 -a1 -c21 -g1 -b packed:21 -d packed python3 -u cnn.py
