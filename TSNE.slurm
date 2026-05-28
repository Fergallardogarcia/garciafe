#!/bin/bash
#SBATCH -A NAISS2025-22-1228 -p alvis
#SBATCH -t 0-02:30:00
#SBATCH --gpus-per-node=V100:1
#SBATCH -N 1 --ntasks=1
#SBATCH -J TSNE 
#SBATCH --array=0
#SBATCH --output "temp/logs/%x_%a.out"
#SBATCH --error "temp/logs/%x_%a.err"


EXP_TYPE="TSNE" #
SUBSCRIPT="more_samples_" #["", "_larger_batch_size"]

#Change project name

# files=(temp/CIFAR10/configs/PCGRAD/*$EXP_TYPE*.yaml)
files=(c-GAN_code/configs/*$EXP_TYPE*.yaml)
CONFIG_FILE=${files[$SLURM_ARRAY_TASK_ID]}

RUN_RESULTS="model_weights"

SAVE_CLASS_GRIDS=1
GRID_SAMPLES_PER_CLASS=4
SKIP_DIS_GEN_EVAL=0

EXTRA_ARGS=""
if [[ "$SAVE_CLASS_GRIDS" == "1" ]]; then
	EXTRA_ARGS+=" --save-class-grids --grid-samples-per-class ${GRID_SAMPLES_PER_CLASS}"
fi
if [[ "$SKIP_DIS_GEN_EVAL" == "1" ]]; then
	EXTRA_ARGS+=" --skip-dis-gen-eval"
fi

srun ./deploy_tsne.sh "$CONFIG_FILE" "$RUN_RESULTS" --eval-only $EXTRA_ARGS


