#!/bin/bash
#SBATCH -A TRN016
#SBATCH -J 08_omp
#SBATCH -o %x-%j.out
#SBATCH -t 0:10:00
#SBATCH -p batch
#SBATCH -N 1

module load DefApps
module load PrgEnv-amd

export OMP_NUM_THREADS=8

srun -N 1 -n 1 -c 8 --threads-per-core=1 ./run
