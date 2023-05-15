#!/bin/bash
#SBATCH -A TRN016
#SBATCH -J mat_mul
#SBATCH -o %x-%j.out
#SBATCH -t 0:10:00
#SBATCH -p batch
#SBATCH -N 1


module load PrgEnv-amd
module load craype-accel-amd-gfx90a
module load openblas

srun -n1 -c1 --gpus-per-task=1 --gpu-bind=closest ./matrix_multiply
