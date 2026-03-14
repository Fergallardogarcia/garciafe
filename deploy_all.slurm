#!/bin/bash
#SBATCH -A NAISS2025-22-1228 -p alvis
#SBATCH -t 0-04:00:00
#SBATCH --gpus-per-node=V100:2
#SBATCH -N 1 --ntasks=1
#SBATCH -J control
#SBATCH --array=321
#SBATCH --output "temp/logs/%x_%a.out"
#SBATCH --error "temp/logs/%x_%a.err"


EXP_TYPE="control" #["less_strength", "more_samples", "control"]
SUBSCRIPT="0"


CONFIG_FILE="temp/CIFAR10/configs/${EXP_TYPE}/${EXP_TYPE}_${SUBSCRIPT}${SLURM_ARRAY_TASK_ID}.yaml"

srun ./deploy.sh "$CONFIG_FILE"
