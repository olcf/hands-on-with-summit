#!/bin/bash
#SBATCH -A TRN016
#SBATCH -J hello
#SBATCH -o %x-%j.out
#SBATCH -t 0:10:00
#SBATCH -p batch
#SBATCH -N 1


srun -n4 -c1 ./run
