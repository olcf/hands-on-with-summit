#!/bin/bash
#SBATCH -A TRN016
#SBATCH -J pytorch_cnn
#SBATCH -o %x-%j.out
#SBATCH -t 1:00:00
#SBATCH -p batch
#SBATCH -N 1

unset SLURM_EXPORT_ENV

date

module load PrgEnv-gnu
module load amd-mixed/5.6.0
module load craype-accel-amd-gfx90a

export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

source ~/miniconda-frontier-handson/bin/activate base
source activate ~/.conda/envs/torch-frontier

python3 -u cnn.py
