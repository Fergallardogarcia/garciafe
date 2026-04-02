#!/bin/bash
#SBATCH -A NAISS2025-22-1228 -p alvis
#SBATCH -t 0-04:00:00
#SBATCH --gpus-per-node=V100:2
#SBATCH -N 1 --ntasks=1
#SBATCH -J s_s_diffArch
#SBATCH --array=0-6
#SBATCH --output "temp/logs/%x_%a.out"
#SBATCH --error "temp/logs/%x_%a.err"


EXP_TYPE="more_samples" #["less_strength", "more_samples", "control"]
SUBSCRIPT="more_samples_" #["", "_larger_batch_size"]

#Change projec name

files=(temp/CIFAR10/configs/more_samples/*more_samples_0*.yaml)
CONFIG_FILE=${files[$SLURM_ARRAY_TASK_ID]}


srun ./deploy.sh "$CONFIG_FILE"


# NAISS2025-22-1228
# Berzelius-2026-100