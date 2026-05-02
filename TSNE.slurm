#!/bin/bash
#SBATCH -A NAISS2025-22-1228 -p alvis
#SBATCH -t 0-02:30:00
#SBATCH --gpus-per-node=T4:1
#SBATCH -N 1 --ntasks=1
#SBATCH -J TSNE 
#SBATCH --array=0
#SBATCH --output "temp/logs/%x_%a.out"
#SBATCH --error "temp/logs/%x_%a.err"


EXP_TYPE="TSNE" #
SUBSCRIPT="more_samples_" #["", "_larger_batch_size"]

#Change project name

files=(temp/CIFAR10/configs/PCGRAD/*$EXP_TYPE*.yaml)
CONFIG_FILE=${files[$SLURM_ARRAY_TASK_ID]}



srun ./deploy_tsne.sh "$CONFIG_FILE"


# NAISS2025-22-1228
# Berzelius-2026-100