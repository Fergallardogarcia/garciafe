"""Module to run a single experiment on slurm like environment."""

import warnings
warnings.filterwarnings("ignore")

import multiprocessing
from logging import DEBUG

import copy
import torch
import os
from os.path import join
import argparse
import ntpath

import fedml
from fedml.common import log

from fedml.client import create_client
from fedml.configs import parse_configs
from fedml.data_handler import load_and_fetch_split, merge_splits
from fedml.models import load_model
from fedml.modules import ExperimentManager, setup_random_seeds
from fedml.server import (
    create_server,
    get_client_manager
)
from fedml.strategy import get_strategy


def single_node_simulation(exp_name, user_configs, executor_type, num_gpus=None):

    # Extract required user configurations
    total_clients = user_configs["SERVER_CONFIGS"]["MIN_NUM_CLIENTS"]
    min_sample_size = user_configs["SERVER_CONFIGS"]["MIN_TRAINING_SAMPLE_SIZE"]

    # Get run device information
    run_devices = None
    if (user_configs["CLIENT_CONFIGS"]["RUN_DEVICE"] == "auto") and (num_gpus is not None):
        # run_devices = [f"cuda:{i%num_gpus}" for i in range(min_sample_size)]
        run_devices = [f"cuda:{i%num_gpus}" for i in range(total_clients)]
    else:
        # run_devices = [user_configs["CLIENT_CONFIGS"]["RUN_DEVICE"] for i in range(min_sample_size)]
        run_devices = [user_configs["CLIENT_CONFIGS"]["RUN_DEVICE"] for i in range(total_clients)]

    log(DEBUG, f"Run device: {run_devices[0]}")

    # Load all dataset and make splits 
    (train_splits, split_labels), testset = load_and_fetch_split(n_clients=total_clients, dataset_conf=user_configs["DATASET_CONFIGS"])

    # Load appropriate number of local models
    # local_models = [load_model(model_configs=user_configs["MODEL_CONFIGS"]).to(run_devices[i]) for i in range(min_sample_size)]
    local_models = [load_model(model_configs=user_configs["MODEL_CONFIGS"]).to(run_devices[i]) for i in range(total_clients)]

    # Create client objects based on client types
    mal_client_type = user_configs["EXPERIMENT_CONFIGS"]["MAL_CLIENT_TYPE"]
    num_mal_clients = int(user_configs["EXPERIMENT_CONFIGS"]["MAL_CLIENT_FRAC"] * total_clients)
    num_hon_clients = total_clients - num_mal_clients

    # Setup clients with different types
    log(DEBUG, f"Creating {num_hon_clients} honest clients and {num_mal_clients} malicious clients of type {mal_client_type}.")
    clients = [create_client(None, id, trainset=train_splits[id], testset=testset, process=(executor_type=="ProcessPool"), configs=user_configs["EXPERIMENT_CONFIGS"]) for id in range(num_hon_clients)]
    
    if user_configs["EXPERIMENT_CONFIGS"]["MAL_SHARED_DATA"]:
        # Merge train_splits reserved for malicious clients
        merged_trainset = merge_splits(train_splits[num_hon_clients:])
        clients.extend([
            create_client(mal_client_type, id+num_hon_clients, trainset=copy.deepcopy(merged_trainset), testset=testset, process=(executor_type=="ProcessPool"), configs=user_configs["EXPERIMENT_CONFIGS"]) 
            for id in range(num_mal_clients)
        ])
    else:
        clients.extend([
            create_client(mal_client_type, id+num_hon_clients, trainset=train_splits[id+num_hon_clients], testset=testset, process=(executor_type=="ProcessPool"), configs=user_configs["EXPERIMENT_CONFIGS"]) 
            for id in range(num_mal_clients)
        ])


    ###########################################################
    ###########################################################
    # Setup a Federated Server instance
    ###########################################################
    ###########################################################

    # Fetch stats and store them locally?
    exp_manager = ExperimentManager(experiment_id=exp_name, hyperparameters=user_configs)

    # Create aggregation strategy
    agg_strat = get_strategy(local_models, run_devices, user_configs=user_configs)

    # Create a client manager
    client_manager = get_client_manager(user_configs=user_configs)

    # Register all clients with client_manager
    for client in clients: 
        client_manager.register(client=client)
    log(
        DEBUG, 
        f"Successfully registered {client_manager.num_available()} clients..."
    )

    # Load pre-trained weights if any are provided
    if "WEIGHT_PATH" in user_configs["MODEL_CONFIGS"].keys() and user_configs["MODEL_CONFIGS"]["WEIGHT_PATH"] is not None:
        local_models[0].load_state_dict(torch.load(user_configs["MODEL_CONFIGS"]["WEIGHT_PATH"], weights_only=False))
    initial_parameters = local_models[0].get_weights()
    if executor_type=="ProcessPool": initial_parameters = initial_parameters.cpu()

    # Create the server instance
    fedml_server = create_server(
        server_type=user_configs["SERVER_CONFIGS"]["SERVER_TYPE"],
        client_manager=client_manager,
        strategy=agg_strat,
        user_configs=user_configs,
        experiment_manager=exp_manager,
        initial_parameters=initial_parameters,
        executor_type=executor_type,
        num_gpus=num_gpus,
    )

    # Train the server for specified number of rounds
    history, runtime = fedml_server.fit(num_rounds=user_configs["SERVER_CONFIGS"]["NUM_TRAIN_ROUND"])

    # Save logging results to disk
    log(
        DEBUG, 
        "Saving logged results to disk ..."
    )
    exp_manager.save_to_disc(user_configs["OUTPUT_CONFIGS"]["RESULT_LOG_PATH"], exp_name)
    history.save_to_disc(user_configs["OUTPUT_CONFIGS"]["RESULT_LOG_PATH"], exp_name)

    # Save the final model state to disk
    log(
        DEBUG, 
        "Saving final model parameters ..."
    )
    torch.save(obj=fedml_server.parameters, f=user_configs["OUTPUT_CONFIGS"]["RESULT_LOG_PATH"] + f"weights-{exp_name}.pt")

    log(
        DEBUG, 
        f"Finished federated experiment {exp_name}.yaml ...\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Run experiment for given configuration file.")
    parser.add_argument(
        "--num-gpus",
        type=int,
        required=True,
        help="Number of Allocated GPUs (no default)",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Configuration file path (no default)",
    )
    parser.add_argument(
        "--executor-type",
        type=str,
        default="ProcessPool",       # ThreadPool, ProcessPool
        help="Run clients on thread or process pool (default: ThreadPool)",
    )
    args = parser.parse_args()

    user_configs = parse_configs(args.config_file)
    exp_name = ntpath.basename(args.config_file)[:-5]

    # Setup random seeds before anything else
    setup_random_seeds(seed_value=user_configs["SERVER_CONFIGS"]["RANDOM_SEED"])

    # Create stdout re-direction files
    os.makedirs(user_configs["OUTPUT_CONFIGS"]["RESULT_LOG_PATH"], exist_ok=True)
    logfile = open( join(user_configs["OUTPUT_CONFIGS"]["RESULT_LOG_PATH"], f"console_{exp_name}.log"), "w")
    fedml.common.logger.update_console_handler(level=DEBUG, stream=logfile)

    # Setup default torch device
    default_device = "cpu"
    if torch.cuda.is_available(): # and args.executor_type == "ThreadPool":
        default_device = f"cuda:{torch.cuda.current_device()}"
    torch.set_default_device(default_device)

    log(DEBUG, f"# of GPUs       : {args.num_gpus}")
    log(DEBUG, f"Config File     : {args.config_file}")
    log(DEBUG, f"Executor Type   : {args.executor_type}")

    single_node_simulation(exp_name=exp_name, user_configs=user_configs, num_gpus=args.num_gpus, executor_type=args.executor_type)

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
