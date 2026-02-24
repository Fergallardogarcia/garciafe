#!/bin/bash

# # User vairables
# dirchlet_alpha=0.5
# split_strategy="dirichlet"
# client_data_path="./data/clients"
# client_data_file="mnist.pt"

# Fetch task array, and process related information
# current_proc_num=$SLURM_LOCALID
# num_parallel_procs=$SLURM_NTASKS

# # Setup configuration variables
# environment_path= "./containers/fl_env_v5.sif" #"/mimer/NOBACKUP/groups/naiss2023-22-904/env_containers/youssef/fedn_container.sif"

# # Client run command
# FEDN_API_URL="https://api.fedn.scaleoutsystems.com/<PROJ-LINK>"
# FEDN_TOKEN="<TOKEN>"
# FEDN_COMMAND="python client/client_runner.py --api-url ${FEDN_API_URL} --token ${FEDN_TOKEN} --name client-${current_proc_num} --data-path ${client_data_path}/$((${current_proc_num}+1))/${client_data_file}"

# Use the apptainer to run the federated experiments.
# echo "Process: ${current_proc_num} of ${num_parallel_procs}"

# # Split datase
# if [ "${current_proc_num}" -eq "0" ]; then
#     apptainer exec --env "FEDN_DIRICHLET_ALPHA=${dirchlet_alpha}" --env "FEDN_DATA_SPLITTING_STRATEGY=${split_strategy}" --env "FEDN_DATA_PATH=${client_data_path}" --env "FEDN_NUM_DATA_SPLITS=${num_parallel_procs}" --nv $environment_path python client/data.py
# else
#     sleep 30
# fi

# Run the clients
# apptainer exec --env "FEDN_PACKAGE_EXTRACT_DIR=package" --env "FEDN_NUM_DATA_SPLITS=${num_parallel_procs}" --env "FEDN_DIRICHLET_ALPHA=${dirchlet_alpha}" --env "FEDN_DATA_SPLITTING_STRATEGY=${split_strategy}" --env "FEDN_DATA_PATH=${client_data_path}" --nv $environment_path ${FEDN_COMMAND}



    
# Fetch task array, and process related information
    #cd /cephyr/users/garciafe
current_proc_num=$SLURM_LOCALID
num_parallel_procs=$SLURM_NTASKS

# Setup configuration variables
environment_path="/cephyr/users/garciafe/containers/fl_env_v5.sif"
NUM_GPUS=${SLURM_GPUS_ON_NODE:-0}
code_path="/cephyr/users/garciafe/c-GAN_code"
CONFIG_FILE="/configs/base_modified.yaml"
FEDN_COMMAND="python fedml/run_federated.py --num-gpus ${NUM_GPUS} --config-file ${CONFIG_FILE}"


echo "GPUs allocated: $NUM_GPUS"
echo "Process: ${current_proc_num} of ${num_parallel_procs}"
apptainer exec --nv $environment_path ${FEDN_COMMAND}
