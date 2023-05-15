#!/bin/bash
#SBATCH -A TRN016
#SBATCH -J profiling_optimized
#SBATCH -o %x-%j.out
#SBATCH -N 1
#SBATCH -t 00:15:00

module load PrgEnv-amd

srun -n1 -c1 --gpus-per-task=1 rocprof --stats  -o metrics_matrix_sums_optimized.csv ./matrix_sums_optimized


# May not be needed anymore.
# OUTPUT=/lustre/orion/TRN016/scratch/${USER}	
# srun -n1 -c1 --gpus-per-task=1 rocprof --stats  -o $OUTPUT/metrics_matrix_sums_optimized.csv ./matrix_sums_optimized
# bug with -i flag means this won't produce correct timing output
# srun -n1 -c1 --gpus-per-task=1 rocprof --stats -i metrics_matrix_sums_kernels.txt -o $OUTPUT/metrics_matrix_sums_optimized.csv ./matrix_sums_optimized