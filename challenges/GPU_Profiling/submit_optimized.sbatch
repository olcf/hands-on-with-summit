#!/bin/bash
#SBATCH -A TRN016
#SBATCH -J profiling_optimized
#SBATCH -o %x-%j.out
#SBATCH -N 1
#SBATCH -t 00:15:00

module load PrgEnv-amd

OUTPUT=/lustre/orion/TRN016/scratch/${USER}	

# bug with -i flag means this won't produce correct timing output
#srun -n1 -c1 --gpus-per-task=1 rocprof --stats -i metrics_matrix_sums_kernels.txt -o $OUTPUT/metrics_matrix_sums_optimized.csv ./matrix_sums_optimized

srun -n1 -c1 --gpus-per-task=1 rocprof --stats  -o $OUTPUT/metrics_matrix_sums_optimized.csv ./matrix_sums_optimized

cp $OUTPUT/metrics_matrix_sums_optimized.*csv $SLURM_SUBMIT_DIR