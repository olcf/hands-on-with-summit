#!/bin/bash
#BSUB -P TRN022
#BSUB -W 00:05
#BSUB -nnodes 1
#BSUB -J mpi4py
#BSUB -o mpi4py.%J.out
#BSUB -e mpi4py.%J.err

cd $LSB_OUTDIR
date

module load gcc
module load hdf5
module load python

source activate $HOME/.conda/envs/h5pympi-ascent

jsrun -n1 -r1 -a42 -c42 python3 hello_mpi.py
