"""Implementation of Honest Client using FedML Framework"""

import copy
from fedml.defenses.filters.gan_filter import generate_dataset

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

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
from models.model import BaseModel
from models.model_loader import load_model 

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
            gen_model: Optional[BaseModel] = None,
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
        self.scale_factor = 1
        if gen_model is None:
            self.gen_model = load_model(attack_config["GAN_ATTACK_CONFIG"]["HYPER_PARAM"]["GEN_ARGS"])
        else:
            self.gen_model = gen_model
        '''
        Perhaps insetad of initializing a malicious generator here, we can initialize it at the server and pass it to the client during training. This way we can avoid initializing a generator for every malicious client and instead just have one generator at the server that is used by all malicious clients. This would also allow us to have a more consistent attack across all malicious clients.

        Also, maybe we just need to pass the generator model class to the client and then initialize it at the client using the global weights received from the server. This way we can avoid having to pass the generator model class from the server to the client during training and instead just have it initialized at the client using the global weights received from the server. This would also allow us to have a more consistent attack across all malicious clients.
        '''

    @property
    def client_type(self):
        """Returns current client's type."""
        return "GAN_ATTACK"


    
    ##########################

    #Generate D_syn

    # Generate a new validation dataset
    

    def generate_D_mixed(
            self,
            gen_model       ,#=self.gen_model,
            attack_config   ,#=self.attack_config,
            # input_znoises   ,#=self.input_znoises,
            # input_classes   ,#=self.input_classes,
            device          ,#= self.device,
            batch_size=1024, # Adjust to have same number of batches as trainset
            server_round: int,
            trainset: Dataset, 
            generate_config: Optional[Dict],
            ) -> Dataset:
        """ 1. Generate synthetic data for client according to trainset distribution:
                Option a: (Same number for each class) (This might exploit c-GAN evaluation better). 
                Option b: (Data according to trainset distribution).
            2. Also, generate either:
                a. Same number of data points as trainset 
                b. Many more, later adjusting per sample loss to account for this (parameter alpha).  

            generate_config: 
                "D_syn_size": Int (Number) or Str ("Train_SIZE")
                "Proportion_per_class": Str ("Equal") or ("Data_like")
        
        return Dataset, alpha (the ratio of synthetic data to real data)
        """ 

        gen_configs= attack_config["GAN_ATTACK_CONFIG"]["HYPER_PARAM"]["GEN_ARGS"]
        num_classes= gen_configs["NUM_CLASSES"]
        latent_size = gen_configs["LATENT_SIZE"]
        
        num_samples = len(trainset) # for now same gen_configs["SYTHETIC_DATA_SIZE"]


        input_znoises = torch.randn(num_samples, latent_size).to(device)
        input_classes = (torch.arange(num_samples)%num_classes).to(device) # By chatgpt


        gen_dataset = generate_dataset(
            gen_model=gen_model,
            input_znoises=input_znoises,
            input_classes=input_classes,
            device=device,
            batch_size=batch_size, 
        )

        data_X_syn, data_Y_syn = gen_dataset.tensors #load in batches later
        data_X_real, data_Y_real = trainset.tensors

        is_real_syn = torch.zeros(len(data_X_syn), dtype=torch.bool)
        is_real_real = torch.ones(len(data_X_real), dtype=torch.bool)

        data_mixed_X = torch.cat((data_X_real, data_X_syn), dim=0).to(device)
        data_mixed_Y = torch.cat((data_Y_real, data_Y_syn), dim=0).to(device)
        is_real = torch.cat((is_real_real, is_real_syn), dim=0).to(device)
        trainset_mixed = TensorDataset(data_mixed_X, data_mixed_Y, is_real).to(device)

        alpha = len(trainset) # Adjusting for different dataset sizes, if needed. 
        return trainset_mixed, alpha
    
    def combined_loss(self, criterion, D_syn_batch: torch.Tensor, D_real_batch: torch.Tensor, combine_config: Optional[Dict], device) -> torch.Tensor:
        """ Combine the generator loss and the local loss to create a new loss function for training the local model. 
            combine_config:
                "Combine_Method": Str ("Weighted_Sum") or ("Other_Method")
                "Gen_Loss_Weight": Float (0-1) (Only if Combine_Method is Weighted_Sum)
        """

        X_mixed_batch, Y_mixed_batch, is_real_batch = D_syn_batch

        
        g_loss_attack = criterion(D_real_batch).to(device)
        return g_loss_attack
        

    def fit(self, model, device, ins: FitIns) -> FitRes:
        # print(f"[Client {self.client_id}] fit, config: {ins.config}")


        model.set_weights(ins.parameters, clone=(not self._process))                    # Added
        gen_model= self.gen_model.to(device)                                            # Added


        

        # Will have to modify train()
        # for i, data in enumerate(trainloader)...:

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
