#!/bin/bash
#SBATCH -A TRN016
#SBATCH -J galaxy
#SBATCH -o %x-%j.out
#SBATCH -t 0:10:00
#SBATCH -p batch
#SBATCH -N 1

date

module load PrgEnv-gnu
module load hdf5

source activate /ccs/home/${USER}/.conda/envs/cupy-frontier

srun -n8 python3 data_transfer.py