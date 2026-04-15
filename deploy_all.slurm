#!/bin/bash
#SBATCH -A NAISS2025-22-1228 -p alvis
#SBATCH -t 0-04:30:00
#SBATCH --gpus-per-node=V100:2
#SBATCH -N 1 --ntasks=1
#SBATCH -J alternating
#SBATCH --array=0-1
#SBATCH --output "temp/logs/%x_%a.out"
#SBATCH --error "temp/logs/%x_%a.err"


EXP_TYPE="alternate" #["PCGRAD", "more_samples", "control"]
SUBSCRIPT="more_samples_" #["", "_larger_batch_size"]

#Change project name

files=(temp/CIFAR10/configs/PCGRAD/*$EXP_TYPE*.yaml)
CONFIG_FILE=${files[$SLURM_ARRAY_TASK_ID]}


srun ./deploy.sh "$CONFIG_FILE"


# NAISS2025-22-1228
# Berzelius-2026-100