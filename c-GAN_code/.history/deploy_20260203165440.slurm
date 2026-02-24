#!/bin/bash
#SBATCH -A NAISS2024-22-1204 -p alvis
#SBATCH -t 0-01:00:00
#SBATCH --gpus-per-node=T4:1
#SBATCH -N 1 --ntasks=4
#SBATCH -J fedn_client_cifar10
#SBATCH --output "temp/logs/%x_%a_%A.out"
#SBATCH --error "temp/logs/%x_%a_%A.err"

wait
srun ./deploy.sh
wait