#!/bin/bash
#SBATCH -A TRN016
#SBATCH -J vec_add
#SBATCH -o %x-%j.out
#SBATCH -t 0:10:00
#SBATCH -p batch
#SBATCH -N 1


date
echo " " 

module load gcc

export OMP_NUM_THREADS=4

srun -n1 -c64 --threads-per-core=1 ./vec_add

