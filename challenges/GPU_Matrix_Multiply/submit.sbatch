#!/bin/bash
#SBATCH -A TRN016
#SBATCH -J dgemm
#SBATCH -o %x-%j.out
#SBATCH -t 0:10:00
#SBATCH -p batch
#SBATCH -N 1


module load DefApps
module load PrgEnv-amd
module load craype-accel-amd-gfx90a
module load openblas


srun -n1 -c1 --gpus-per-task=1 --gpu_bind=closest ./cpu_gpu_dgemm
