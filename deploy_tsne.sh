#!/bin/bash

# Fetch task array, and process related information
current_proc_num=$SLURM_LOCALID
num_parallel_procs=$SLURM_NTASKS

# Setup configuration variables
environment_path="/cephyr/users/garciafe/containers/fl_env_v5.sif"
NUM_GPUS=${SLURM_GPUS_ON_NODE:-0}
base_folder="/cephyr/users/garciafe"
code_path="${base_folder}/c-GAN_code"
CONFIG_FILE="${1:-./configs/base_2.yaml}"

# Command to run inside container
TSNE_COMMAND="python ${code_path}/fedml/tSNE.py --num-gpus ${NUM_GPUS} --config-file ${base_folder}/${CONFIG_FILE} --executor-type ProcessPool"

echo "GPUs allocated: $NUM_GPUS"
echo "Process: ${current_proc_num} of ${num_parallel_procs}"
NUM_GPUS=${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-0}}
echo "SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE"
echo "SLURM_GPUS_PER_NODE=$SLURM_GPUS_PER_NODE"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"

# Run with PYTHONPATH correctly set
apptainer exec --nv --pwd ${code_path} --env PYTHONPATH=${code_path} $environment_path $TSNE_COMMAND
