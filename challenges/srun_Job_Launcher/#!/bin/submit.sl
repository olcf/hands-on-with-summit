#!/bin/bash
#SBATCH -A STF007  
#SBATCH -J testing_srun
#SBATCH -o %x-%j.out
#SBATCH -t 0:10:00
#SBATCH -p batch
#SBATCH -N 1

# ************************
# ONLY EDIT FROM HERE DOWN
# ************************

# number of OpenMP threads
export OMP_NUM_THREADS=1

# jsrun command to modify (see key above for flags)
srun -N 1 -n 1 -c 1 -gpus 1 | sort
