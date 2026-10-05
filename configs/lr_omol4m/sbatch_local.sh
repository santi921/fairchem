#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -c 32
#SBATCH -q regular
#SBATCH -J lr_omol4m
#SBATCH --mail-user=santiagovargas@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -A m3278_g
#SBATCH -t 24:0:0
# Fallback to submit.sh: run one arm inside a plain sbatch allocation using
# fairchem's LOCAL launcher (torch elastic, 4 ranks). Unlike SLURM mode this
# does not requeue on timeout; resume manually from the run's checkpoint.
#
# Usage:
#   sbatch configs/lr_omol4m/sbatch_local.sh configs/lr_omol4m/uma_direct.yaml uma_lr=coulomb
set -euo pipefail

module load python
conda activate /pscratch/sd/s/santiago/envs/fairchem_lr

export OMP_NUM_THREADS=8
srun --ntasks=1 --gpus=4 fairchem -c "$@" cluster.mode=LOCAL
