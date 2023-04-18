#!/bin/bash 
#SBATCH -A TRN016
#SBATCH -J add_vec_cpu
#SBATCH -o %x-%j.out
#SBATCH -t 00:10:00
#SBATCH -p batch
#SBATCH -N 1

srun -n1 --ntasks-per-node=1 ./run 
