#!/bin/bash
#SBATCH -A TRN016
#SBATCH -J ptp_hello
#SBATCH -o %x-%j.out
#SBATCH -t 0:10:00
#SBATCH -p batch
#SBATCH -N 1


srun -n2 -c1 ./run                            
