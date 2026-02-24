#!/bin/bash
#SBATCH -A NAISS2025-22-1228 -p alvis
#SBATCH -t 0-02:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH -N 1 --ntasks=1
#SBATCH -J trial_fer
#SBATCH --output "temp/logs/%x_%A.out"
#SBATCH --error "temp/logs/%x_%A.err"

wait
srun ./deploy.sh
wait

# --gpus-per-node=T4:2 --gpus-per-node=A40:2