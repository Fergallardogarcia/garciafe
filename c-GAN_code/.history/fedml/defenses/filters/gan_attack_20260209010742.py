
from typing import Dict, List, Tuple, Optional
from logging import INFO, DEBUG
from collections import defaultdict

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import scipy.stats
from sklearn.cluster import KMeans

from fedml.common import Parameters, log
from fedml.models import load_model
from fedml.modules import get_optimizer, get_criterion, train_generator, evaluate_gan

from .filter import Filter
from .gan_filter import GenerativeFilter


class MaliciousGenerator():
    def __init__(self, gen_configs, train_configs, filter_configs, skip_rounds = 1) -> None:
        self.gen_configs = gen_configs
        # self.dis_configs = dis_configs
        self.train_configs = train_configs
        # self.filter_configs = filter_configs

        # Setup models (both generator and discriminator)
        self.gen_model = load_model(model_configs=gen_configs)
        # self.dis_model = load_model(model_configs=dis_configs) # model is to be provided by the server during training

        self.criterion = get_criterion(criterion_str=train_configs["CRITERION"])

        self.device = train_configs["DEVICE"]
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            # torch.get_default_device()
        
        # Setup random data needed for dataset generation
        
        # Use trained generator to create samples
        self.num_classes = gen_configs["NUM_CLASSES"]
        # self.samples_per_class = filter_configs["SAMPLES_PER_CLASS"]
        self.latent_size = gen_configs["LATENT_SIZE"]
        self.skip_rounds = skip_rounds

        # Generate input randomness
        self.input_znoises = torch.randn(self.num_classes * self.samples_per_class, self.latent_size)
        self.input_classes = torch.arange(self.num_classes, dtype=torch.int64).repeat_interleave(self.samples_per_class)

    
    def server_tasks(self, model, server_round: int) -> None: #server task inputs model instead of global weights to keep consistency with other clients
        # Skip first round as we don't have a baseline to perform comparison
        if server_round < self.skip_rounds:
            log(
                INFO,
                "attack_updates: Skipping attack at server round %s",
                server_round,
            )
            return self

        # # Set new weights to the generator
        # self.dis_model.set_weights(weights=global_weights)

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
            dis_model=model,
            gen_optim=gen_optimizer,
            criterion=self.criterion,
            iterations=self.train_configs["ITERATION"],
            latent_size=self.latent_size,
            num_classes=self.num_classes,
            batch_size=self.train_configs["BATCH_SIZE"],
            device=self.device
        )

        # Finished training return
        return self



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


