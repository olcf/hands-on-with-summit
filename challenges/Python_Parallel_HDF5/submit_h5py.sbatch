#!/bin/bash
#SBATCH -A TRN016
#SBATCH -J h5py
#SBATCH -o %x-%j.out
#SBATCH -t 0:10:00
#SBATCH -p batch
#SBATCH -N 1

unset SLURM_EXPORT_ENV

date

module load PrgEnv-gnu
module load hdf5

source ~/miniconda-frontier-handson/bin/activate base
source activate ~/.conda/envs/h5pympi-frontier

srun -n42 python3 -u hdf5_parallel.py
