#!/bin/bash
#SBATCH -A NAISS2025-22-1228 -p alvis
#SBATCH -t 0-05:00:00
#SBATCH --gpus-per-node=V100:1
#SBATCH -N 1 --ntasks=1
#SBATCH -J trial_fer
#SBATCH --array=7-11
#SBATCH --output "temp/logs/%x_%a.out"
#SBATCH --error "temp/logs/%x_%a.err"

CONFIG_FILE="./configs/base_${SLURM_ARRAY_TASK_ID}.yaml"

srun ./deploy.sh "$CONFIG_FILE"
