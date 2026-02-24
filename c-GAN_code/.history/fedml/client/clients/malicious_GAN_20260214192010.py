"""Implementation of Honest Client using FedML Framework"""

import copy
from fedml.defenses.filters.gan_attack import generate_dataset
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
            gen_model: Optional[BaseModel] = None
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
    

    def generate_D_syn(
            self,
            gen_model=self.gen_model,
            input_znoises=self.input_znoises,
            input_classes=self.input_classes,
            device=self.device,
            batch_size=1024,
            updated_generator_model: torch.Tensor, server_round: int,
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
        
        return Dataset, 
        """ 
        gen_dataset = generate_dataset(
        gen_model=self.gen_model,
        input_znoises=self.input_znoises,
        input_classes=self.input_classes,
        device=self.device,
        batch_size=1024,
    )

        return Dataset, alpha
    
    def combined_loss(self, criterion, D_syn: torch.Tensor, D_real: torch.Tensor, combine_config: Optional[Dict]) -> torch.Tensor:
        """ Combine the generator loss and the local loss to create a new loss function for training the local model. 
            combine_config:
                "Combine_Method": Str ("Weighted_Sum") or ("Other_Method")
                "Gen_Loss_Weight": Float (0-1) (Only if Combine_Method is Weighted_Sum)
        """
        
        """Requisites:

        criterion = modules.get_criterion(
            criterion_str=criterion_str
        )
        optimizer = modules.get_optimizer(
            optimizer_str=optimizer_str,            
            local_model=model,
            learning_rate=learning_rate,
            **optim_kwargs,
        )

        num_examples = modules.train(
            model=model, 
            trainloader=trainloader, 
            epochs=local_epochs, 
            learning_rate=learning_rate,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )
        
        """


        g_loss_attack = criterion(dis_predict, input_l).to(device)
        return
        

    def fit(self, model, device, ins: FitIns) -> FitRes:
        # print(f"[Client {self.client_id}] fit, config: {ins.config}")


        model.set_weights(ins.parameters, clone=(not self._process))                    # Added
        gen_model= self.gen_model.to(device)                                                        # Added

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
