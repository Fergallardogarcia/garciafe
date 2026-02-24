"""Implementation of Honest Client using FedML Framework"""

import copy
import numpy as np

import torch
from torch.utils.data import Dataset

from logging import DEBUG
from typing import Optional, Dict
from fedml.common import (
    FitIns,
    FitRes,
    log
)

from .honest_client import HonestClient
from fedml.defenses.filters.gan_attack import MaliciousGenerator as MalGen #Added

class GanMaliciousClient(HonestClient):
    """A malicious client submitting updates with flipped gradient signs.
    
    """
    def __init__(
            self, 
            client_id: int,
            trainset: Dataset,
            testset: Dataset,
            process: bool = True,
            attack_config: Optional[Dict] = None,
        ) -> None:
        """Initializes a new client."""
        super().__init__(
            client_id=client_id,
            trainset=trainset,
            testset=testset,
            process=process,
        )
        self.attack_config = copy.deepcopy(attack_config)
        filter_type = user_configs["SERVER_CONFIGS"]["FILTER_CONFIGS"]["FILTER_TYPE"]
        filter_param = user_configs["SERVER_CONFIGS"]["FILTER_CONFIGS"]["HYPER_PARAM"]

            if filter_type == "GAN-FILTERING":
         from .filters.gan_filter import GenerativeFilter
        return GenerativeFilter(
            gen_configs=filter_param["GEN_ARGS"],
            dis_configs=user_configs["MODEL_CONFIGS"],
            train_configs=filter_param["TRAIN_GAN_PARAMS"],
            filter_configs=filter_param["FILTER_ARGS"],
            skip_rounds=filter_param["SKIP_ROUNDS"],
        )
        self.malicious_generator = MalGen(self, gen_configs, dis_configs, train_configs, filter_configs, skip_rounds = 1)

    @property
    def client_type(self):
        """Returns current client's type."""
        return "GAN_ATTACK"

    ##########################

    def fit_generator(self, global_weights: torch.Tensor, server_round: int):
        # Start training of generator from current model parameters
        submitted_fs = {
            self.executor.submit(self.malicious_generator.server_tasks, self.parameters, server_round)
        }

    def fit(self, model, device, ins: FitIns) -> FitRes:
        # print(f"[Client {self.client_id}] fit, config: {ins.config}")

        # Don't perform attack until specific round, even
        # then perform with a specified probability.
        server_round = int(ins.config["server_round"])
        attack = np.random.random() < self.attack_config["ATTACK_RATIO"]

        if (server_round < self.attack_config["ATTACK_ROUND"]) or not attack:
            return super().fit(model, device, ins=ins)
        
        fit_results = super().fit(model, device, ins=ins)
        fit_results.metrics["attacking"] = True

        # Flip Gradient Signs
        # Correct Update  : Update = New_Model - Old_Model
        # Flipped Update  : New_Model = Old_Model - Update
        update = fit_results.parameters - ins.parameters
        fit_results.parameters = ins.parameters - self.scale_factor * update

        return fit_results
