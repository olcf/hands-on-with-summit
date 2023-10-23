#!/bin/bash 
#SBATCH -A TRN016
#SBATCH -J add_vec_cpu
#SBATCH -o %x-%j.out
#SBATCH -t 00:10:00
#SBATCH -p batch
#SBATCH -N 1

module load PrgEnv-amd

srun -n1 --cpu-bind=map_cpu:49 --gpus=1 ./run 
