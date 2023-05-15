#!/bin/bash
#SBATCH -A TRN016
#SBATCH -J h5py
#SBATCH -o %x-%j.out
#SBATCH -t 0:10:00
#SBATCH -p batch
#SBATCH -N 1

date

module load PrgEnv-gnu
module load hdf5

source activate /ccs/home/${USER}/.conda/envs/cupy-frontier

srun -n42 python3 hdf5_parallel.py