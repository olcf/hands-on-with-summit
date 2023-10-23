#!/bin/bash
#SBATCH -A TRN016
#SBATCH -J add_vec_acc
#SBATCH -o %x-%j.out
#SBATCH -t 0:10:00
#SBATCH -p batch
#SBATCH -N 1

date

srun -n 1 -c 1 -gpus 1 ./run

#Missing profiler