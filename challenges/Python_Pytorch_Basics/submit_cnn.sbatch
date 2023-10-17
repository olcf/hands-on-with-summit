#!/bin/bash
#SBATCH -A TRN016
#SBATCH -J pytorch_cnn
#SBATCH -o %x-%j.out
#SBATCH -t 1:00:00
#SBATCH -p batch
#SBATCH -N 1

unset SLURM_EXPORT_ENV

date

module load PrgEnv-gnu
module load craype-accel-amd-gfx90a

source activate /ccs/home/${USER}/conda_envs/frontier/py39-frontier

srun -n1 -c21 --gpus=1 --gpu-bind=closest --threads-per-core=1 -m block:block  python3 -u cnn.py
