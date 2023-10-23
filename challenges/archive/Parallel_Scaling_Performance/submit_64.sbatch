#!/bin/bash
#SBATCH -A TRN016
#SBATCH -J 64_omp
#SBATCH -o %x-%j.out
#SBATCH -t 0:10:00
#SBATCH -p batch
#SBATCH -N 1

module load DefApps
module load PrgEnv-amd

export OMP_NUM_THREADS=64

srun -N 1 -n 1 -c 64 --threads-per-core=1 ./run
