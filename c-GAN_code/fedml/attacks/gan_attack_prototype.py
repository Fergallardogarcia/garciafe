from abc import ABC, abstractmethod
from logging import INFO, DEBUG
from multiprocessing.connection import Client
from typing import Dict, Optional, List, Tuple
from collections import defaultdict
import copy

import os
import threading

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

import concurrent.futures
 

from fedml.modules.strategy_functions import get_fit_config_fn
from fedml.modules import get_optimizer, get_criterion, train_generator, evaluate_gan, train_generator2
from fedml.server import client_manager
from fedml.strategy import Strategy
from fedml.common.typing import Parameters, Scalar, FitIns, GanAttackFitPayload
from fedml.common import log
from fedml.server.client_manager import ClientManager
from fedml.defenses.filters.filter import Filter
from fedml.server.criterion import MaliciousSampling
from fedml.client.clients import malicious_random as malicious_random_attack


from models.model import BaseModel
from models.model_loader import load_model 

from modules.trainer import train_generator

from fedml.data_handler.data_split import CustomDataset

class GAN_attack(Filter):
    """
    def __init__(
        self, 
        *, 
        client_manager, 
        experiment_manager = None,
        strategy = None,
        user_configs: Optional[Dict] = None,
        initial_parameters = None,
        gen_model: BaseModel, # For the moment only one
        executor,
        skip_rounds: int = 1,
        
    ) -> None:
        self.experiment_manager = experiment_manager
        self.user_configs = user_configs
        self.client_manager = client_manager
        self.strategy = strategy
        self.set_initial_parameters(initial_parameters=initial_parameters)
        # self.max_workers: Optional[int] = None
        self.max_workers: Optional[int] = self.strategy.min_fit_clients * 2
        self.executor=executor
        self.gen_model = gen_model
        """

    def __init__(self, 
                 gen_configs, 
                 dis_configs, 
                 train_configs, 
                 filter_configs,
                 cuda_device: Optional[str] = "cuda",
                 skip_rounds = 1,
                 random_seed: int = 0,
                 ) -> None:
            self.gen_configs = gen_configs
            self.filter_configs= filter_configs
            self.dis_configs = dis_configs # Or self.dis_model
            self.train_configs = train_configs
            

             

            self.criterion = get_criterion(criterion_str=train_configs["CRITERION"])

            which_device = train_configs["DEVICE"]
            if which_device in ["auto", "cuda"] and torch.cuda.is_available():
                self.device = cuda_device
            else:
                self.device = "cpu"
            

            # Setup models (both generator and discriminator)
            self.gen_model = load_model(model_configs=gen_configs).to(self.device)
            self.dis_model = load_model(model_configs=dis_configs).to(self.device)

            # Setup random data needed for dataset generation
            
            # Use trained generator to create samples
            self.num_classes = gen_configs["NUM_CLASSES"]
            self.samples_per_class = filter_configs["SAMPLES_PER_CLASS"]
            self.latent_size = gen_configs["LATENT_SIZE"]
            self.skip_rounds = skip_rounds


            # Generate input randomness
            self.input_znoises = torch.randn(self.num_classes * self.samples_per_class, self.latent_size)
            self.input_classes = torch.arange(self.num_classes, dtype=torch.int64).repeat_interleave(self.samples_per_class)



            # Build one fixed perturbation direction once, using the same tensor type as get_weights.
            rng = torch.Generator(device="cpu")
            rng.manual_seed(int(random_seed))
            base_weights = self.dis_model.get_weights()
            direction = malicious_random_attack.rand_like(
                base_weights,
                dtype=base_weights.dtype,
                generator=rng,
            )
            direction_norm = torch.linalg.vector_norm(direction).clamp_min(1e-12)
            self.perturbation_direction = direction / direction_norm

        
            
        

    
    @property
    def filter_type(self) -> str:
        return "GAN_ATTACK"

    # Wraper methods for server hooks-------------------------------
    def server_fit_round_before(
              self, 
              global_parameters, 
              server_round: int, 
              executor,
              client_instructions,
    ):
        # Start training of generator from current model parameters
        submitted_fs = {
            executor.submit(self.train_gen, global_parameters, server_round)
        }

        # Wait until malicious generator is trained
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

        for future in finished_fs:
            trained_weights = future.result()
            if trained_weights is not None:
                self.gen_model.set_weights(trained_weights)

        #Broadcast gen model to clients and return mal_client_ins

        
        self.broadcast_gen_model(client_instructions) #Probably should add security checker for server iteration

        return submitted_fs

    def server_fit_round_after(self):
        return self.eval_gen_attack_round, self.log_gen_attack_stats


    def eval_gen_attack_round(self, server_round: int, results):
        
        
        (mal_client_ids, attack_gen_stats) = self.eval_gen_attack(
            mal_client_weights=[(fit_res.parameters, fit_res.num_examples) for mal_cid, fit_res in results
                                if fit_res.metrics["client_type"] != "HONEST"],
            mal_client_ids=[mal_cid for mal_cid, fit_res in results if fit_res.metrics["client_type"] != "HONEST"],
            server_round=server_round,
            gen_model=self.gen_model
        )

        
        return mal_client_ids, attack_gen_stats


    def eval_gen_attack(
            self, 
            mal_client_weights: List[Tuple[Parameters, int]],
            mal_client_ids: List[int],
            server_round: int,
            gen_model: torch.nn.Module,
            input_znoises: Optional[torch.Tensor] = None,
            input_classes: Optional[torch.Tensor] = None
        ) -> Tuple[List[int], Optional[Dict[str, List]]]:
        """Filter the updates"""
        # Skip first round as we don't have a baseline
        # to perform comparison
        if server_round < self.skip_rounds:
            log(
                INFO,
                "filter_updates: Skipping filteration at server round %s",
                server_round,
            )
            return [indx for indx in range(len(mal_client_weights))], None

        # Generate a new validation dataset
        gen_dataset = generate_dataset(
            gen_model=gen_model,
            input_znoises=input_znoises if input_znoises is not None else self.input_znoises,
            input_classes=input_classes if input_classes is not None else self.input_classes,
            device=self.device,
            batch_size=1024,
        )

        # Perform filteration 
        select_ids, mal_client_stats_attack_gen = perform_evaluation_attack_gen(
            filter_configs=self.filter_configs,
            dis_model=self.dis_model,
            gen_dataset=gen_dataset,
            client_weights=mal_client_weights,
            mal_client_ids=mal_client_ids,
            criterion=self.criterion,
            num_classes=self.num_classes,
            samples_per_class=self.samples_per_class,
            device=self.device
        )

        log(
            DEBUG,
            "Attack gen_updates: accuracies %s",
            mal_client_stats_attack_gen["gen_attack_accu_all"]
        )

        log(
            DEBUG,
            "Attack gen_updates: average loss %s",
            [f"{i:0.4f}" for i in mal_client_stats_attack_gen["gen_attack_loss"]]
        )

        # selected_ids are useless here
        return select_ids, mal_client_stats_attack_gen


    def log_gen_attack_stats(self, metrics_aggregated, mal_client_stats, mal_results):
        # Logging filtering stats of clients
        if self.filter_type == "GAN_ATTACK" and mal_client_stats is not None:
            metrics_aggregated["gen_attack_loss"] = dict()
            metrics_aggregated["gen_attack_accu_all"] = dict()
            metrics_aggregated["gen_attack_accu_cls"] = dict()
            metrics_aggregated["fit_res_metrics"] = dict()
            # Compile / collect results
            
            
            for cid, gen_loss, gen_accu_all, gen_accu_cls in zip(mal_client_stats["client_id"], mal_client_stats["gen_attack_loss"], mal_client_stats["gen_attack_accu_all"], 
                              mal_client_stats["gen_attack_accu_cls"]):
                metrics_aggregated["gen_attack_loss"][f"client_{cid}"] = gen_loss
                metrics_aggregated["gen_attack_accu_all"][f"client_{cid}"] = gen_accu_all
                metrics_aggregated["gen_attack_accu_cls"][f"client_{cid}"] = gen_accu_cls

        return metrics_aggregated



    def train_gen(self, global_weights: List[torch.Tensor], server_round: int) -> None:
        # Skip first round as we don't have a baseline to perform comparison
        if server_round < self.skip_rounds:
            log(
                INFO,
                "GAN_attack_updates: Skipping GAN attack at server round %s",
                server_round,
            )
            return self.gen_model.get_weights()

        # Set new weights to the generator
        self.dis_model.set_weights(weights=global_weights)

        # Create optimizer based on user choice
        gen_optimizer = get_optimizer(
            optimizer_str=self.train_configs["OPTIMIZER"], 
            local_model=self.gen_model, 
            learning_rate=self.train_configs["LEARN_RATE"]
        )

        
        # Start the training process 
        # Train generator model with current
        # global model set as discriminator
        self.gen_model = train_generator(
            gen_model=self.gen_model, 
            dis_model=self.dis_model,
            gen_optim=gen_optimizer,
            criterion=self.criterion,
            iterations=self.train_configs["ITERATION"],
            latent_size=self.latent_size,
            num_classes=self.num_classes,
            batch_size=self.train_configs["BATCH_SIZE"],
            device=self.device
        )

        # Return trained generator weights
        return self.gen_model.get_weights()

    
    
    def select_malicious_clients(self, client_instructions):
        sampling = MaliciousSampling()
        # client_instructions= [(client, self.local_models[client.client_id], self.run_devices[client.client_id], fit_ins)]
        mal_client_ins = [client_ins_n for client_ins_n in client_instructions if client_ins_n[0].client_type == self.filter_type]
        return mal_client_ins

    # Broadcast gen Model (Should only  be done once for now):
    def broadcast_gen_model(self, client_instructions):

        mal_client_ins= self.select_malicious_clients(client_instructions)
        
        client_post_gen_params = None  # Initialize to None before the loop
        server_gen_params = self.gen_model.get_weights()
        for client, _, device, fitins in mal_client_ins:
            if next(client.gen_model.parameters()).device != device:
                log(
                    INFO,
                    "Moving Client GEN model from device: %s, to: %s",
                    next(client.gen_model.parameters()).device,
                    device
                )
                client.gen_model.to(device)
            # Attach Gen models or instructions to the selected malicious clients
            
            client.gen_model.set_weights(server_gen_params) 
            #Client model updated but not in proper device yet. This is done in the fit function.
            client_post_gen_params = client.gen_model.get_weights()
        # Configure the attack for the next round of training here
            # Populate attack payload with perturbation direction
            fitins.gan_attack_payload = GanAttackFitPayload(
                perturbation_direction=self.perturbation_direction
            )

            
        if client_post_gen_params is not None:
            log(
                INFO,
                "Client GEN params are those of server generator: %s",
                (client_post_gen_params == server_gen_params).all()
                )

        return 

def perform_evaluation_attack_gen(
        filter_configs, 
        dis_model, 
        gen_dataset, 
        client_weights,
        mal_client_ids,
        criterion,
        num_classes, 
        samples_per_class, 
        device
    ):
    # Create a dataloader of the generator data
    dataloader = DataLoader(gen_dataset, batch_size=2048, shuffle=False)
    threshold = filter_configs["BASELINE_OVERALL_MIN_ACC"]

    # Perform evaluation of all clients
    select_ids = []
    client_stats = defaultdict(list)
    for index, (weights, _) in zip(mal_client_ids, client_weights):
        # Setup the discriminator model        
        dis_model.set_weights(weights=weights)
        stats = evaluate_gan(
            dis_model=dis_model,
            testloader=dataloader,
            device=device,
            num_classes=num_classes,
            num_sample_per_class=samples_per_class,
            criterion=criterion
        )

        # Log evaluation stats for future reference.
        client_stats["gen_attack_loss"].append(stats[0])
        client_stats["gen_attack_accu_all"].append(stats[1])
        client_stats["gen_attack_accu_cls"].append(stats[2])
        client_stats["client_id"].append(index)
        select_ids.append(index)

    return (select_ids, client_stats)


def generate_dataset(
        gen_model: nn.Module, 
        input_znoises: torch.Tensor, 
        input_classes: torch.Tensor, 
        device: str, 
        batch_size: int,
    ):
    # Stage generator model to the run device
    if next(gen_model.parameters()).device != device:
        gen_model.to(device)

    # Generate sample data using generator model
    data_X = []
    data_Y = []
    gen_model.eval()
    for start_index in range(0, input_classes.size(dim=0), batch_size):
        current_z, current_l = input_znoises[start_index:start_index+batch_size], input_classes[start_index:start_index+batch_size]
        current_z, current_l = current_z.to(device), current_l.to(device)
        # Generate a batch of samples
        data_X.append(gen_model(current_z, current_l).cpu())
        data_Y.append(current_l.cpu())

    # Generate and return a dataset of synthetic data
    data_X, data_Y = torch.vstack(data_X), torch.hstack(data_Y)
    return TensorDataset(data_X, data_Y)