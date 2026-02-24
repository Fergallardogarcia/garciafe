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
from 

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
            gan_model: Optional[model] = None
        ) -> None:
        """Initializes a new client."""
        super().__init__(
            client_id=client_id,
            trainset=trainset,
            testset=testset,
            process=process,
        )
        #--------------------------------------
        self.attack_config = copy.deepcopy(attack_config)
        filter_param = self.attack_config["GAN_ATTACK_CONFIG"]["HYPER_PARAM"]

        if gan_model is None:
            self.malicious_generator = load_model
        else:
            self.gan_model = gan_model
        # MalGen(
        #     gen_configs=filter_param["GEN_ARGS"],
        #     dis_configs=user_configs["MODEL_CONFIGS"],
        #     train_configs=filter_param["TRAIN_GAN_PARAMS"],
        #     filter_configs=filter_param["FILTER_ARGS"],
        #     skip_rounds=filter_param["SKIP_ROUNDS"],
        # )
        '''
        Perhaps insetad of initializing a malicious generator here, we can initialize it at the server and pass it to the client during training. This way we can avoid initializing a generator for every malicious client and instead just have one generator at the server that is used by all malicious clients. This would also allow us to have a more consistent attack across all malicious clients.

        Also, maybe we just need to pass the generator model class to the client and then initialize it at the client using the global weights received from the server. This way we can avoid having to pass the generator model class from the server to the client during training and instead just have it initialized at the client using the global weights received from the server. This would also allow us to have a more consistent attack across all malicious clients.
        '''

    @property
    def client_type(self):
        """Returns current client's type."""
        return "GAN_ATTACK"

    ##########################

    def generate_D_syn(self, global_weights: torch.Tensor, server_round: int):
        # Start training of generator from current model parameters
        submitted_fs = {
            self.executor.submit(self.gan_model.server_tasks, self.parameters, server_round)
        }

    def fit(self, model, device, ins: FitIns) -> FitRes:
        # print(f"[Client {self.client_id}] fit, config: {ins.config}")

        


        model.set_weights(ins.parameters, clone=(not self._process))                    # Added
        self.malicious_generator.server_tasks(model, int(ins.config["server_round"]))   # Added



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
