from abc import ABC, abstractmethod
from logging import INFO, DEBUG
from multiprocessing.connection import Client
from typing import Dict, Optional, List, Tuple

import os
import threading

import torch
import torch.nn as nn


import concurrent.futures
from common.typing import Parameters, Scalar, FitIns

from fedml.modules.strategy_functions import get_fit_config_fn
from fedml.modules import get_optimizer, get_criterion, train_generator, evaluate_gan
from fedml.server import client_manager
from fedml.strategy import Strategy
from fedml.common.typing import Parameters, Scalar
from fedml.common import Parameters, log
from fedml.server.client_manager import ClientManager
from fedml.defenses.filters.filter import Filter
from fedml.server.criterion import MaliciousSampling

from models.model import BaseModel
from models.model_loader import load_model 

from modules.trainer import train_generator

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
                 skip_rounds = 1
                 ) -> None:
            self.gen_configs = gen_configs
            self.dis_configs = dis_configs # Or self.dis_model
            self.train_configs = train_configs
            

             

            self.criterion = get_criterion(criterion_str=train_configs["CRITERION"])

            which_device = train_configs["DEVICE"]
            if which_device in ["auto", "cuda"] and torch.cuda.is_available():
                self.device = cuda_device
                log(
                    INFO,
                    "Using device %s for GAN attack",
                    self.device
                    )
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

        #Broadcast gen model to clients

        
        self.broadcast_gen_model(client_instructions) #Probably should add security checker for server iteration

        return submitted_fs




    def train_gen(self, global_weights: List[torch.Tensor], server_round: int) -> None:
        # Skip first round as we don't have a baseline to perform comparison
        if server_round < self.skip_rounds:
            log(
                INFO,
                "GAN_attack_updates: Skipping GAN attack at server round %s",
                server_round,
            )
            return self

        # Set new weights to the generator
        self.dis_model.set_weights(weights=global_weights)

        # Create optimizer based on user choice
        gen_optimizer = get_optimizer(
            optimizer_str=self.train_configs["OPTIMIZER"], 
            local_model=self.gen_model, 
            learning_rate=self.train_configs["LEARN_RATE"]
        )

        gen_params_prior = self.gen_model.get_weights()
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

        gen_params_post = self.gen_model.get_weights()

        process_id = os.getpid()
        thread_name = threading.current_thread().name

        log(
            INFO,
            "Server attack GEN model updated: %s, device: %s, filter type: %s, process_id: %s, thread_name: %s",

            (gen_params_post != gen_params_prior).any(),
            self.device,
            self.filter_type,
            process_id,
            thread_name
            )


        # Finished training return
        return self

    
    
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
        for client, _, device, _ in mal_client_ins:
            if next(client.gen_model.parameters()).device != device:
                log(
                    INFO,
                    "Moving Client GEN model from device: %s, to: %s",
                    next(client.gen_model.parameters()).device,
                    device
                )
                client.gen_model.to(device)
            # Attach Gen models or instructions to the selected malicious clients
            client_prior_gen_params = client.gen_model.get_weights()
            client.gen_model.set_weights(server_gen_params) 
            #Client model updated but not in proper device yet. This is done in the fit function.
            client_post_gen_params = client.gen_model.get_weights()
        # Configure the attack for the next round of training here 

            log(
                INFO,
                "Client GEN models changed after broadcasting: %s, device: %s",
                (client_prior_gen_params != client_post_gen_params).any(),
                self.device
                )
        if client_post_gen_params is not None:
            log(
                INFO,
                "Client GEN params are those of server generator: %s",
                (client_post_gen_params == server_gen_params).all()
                )

        return

