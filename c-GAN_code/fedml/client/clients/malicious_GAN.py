"""Implementation of Honest Client using FedML Framework"""

import copy
import json
import timeit
from xml.parsers.expat import model
from fedml import modules
from fedml.common.typing import Code
from fedml.defenses.filters.gan_filter import generate_dataset

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

import numpy as np

import torch
from torch.utils.data import Dataset

from logging import DEBUG, INFO
from typing import Optional, Dict
from fedml.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Status,
    log,
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
        self.synth_strength_ratio = self.attack_config["GAN_ATTACK_CONFIG"]["HYPER_PARAM"]["SYNTHETIC_STRENGTH_RATIO"]
        self.num_synth_samples = self.attack_config["GAN_ATTACK_CONFIG"]["HYPER_PARAM"]["SYNTHETIC_DATA_SIZE"]

        if self.num_synth_samples == "trainset_size":
            self.num_synth_samples = len(trainset)
        
        
        if gen_model is None:
            self.gen_model = load_model(attack_config["GAN_ATTACK_CONFIG"]["HYPER_PARAM"]["GEN_ARGS"])
        else:
            self.gen_model = gen_model
        

    @property
    def client_type(self):
        """Returns current client's type."""
        return "GAN_ATTACK"


    
    ##########################

    #Generate D_syn

    # Generate a new validation dataset
    

    def generate_D_mixed(
            self            ,
            gen_model       ,#=self.gen_model,
            attack_config   ,#=self.attack_config,
            # input_znoises   ,#=self.input_znoises,
            # input_classes   ,#=self.input_classes,
            device          ,#= self.device,
            server_round: int,
            trainset: Dataset, 
            generate_config: Optional[Dict]=None,
            batch_size=1024, 
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
        
        num_synth_samples = self.num_synth_samples
        num_real_samples = len(trainset)

        input_znoises = torch.randn(num_synth_samples, latent_size).to(device)
        input_classes = (torch.arange(num_synth_samples)%num_classes).to(device) # By chatgpt

        with torch.no_grad():
            gen_dataset = generate_dataset(
                gen_model=gen_model,
                input_znoises=input_znoises,
                input_classes=input_classes,
                device=device,
                batch_size=batch_size, 
            )

        data_X_syn, data_Y_syn = gen_dataset.tensors #load in batches later
        data_X_real, data_Y_real = trainset.data, trainset.targets

        is_real_syn = torch.zeros(len(data_X_syn), dtype=torch.bool)
        is_real_real = torch.ones(len(data_X_real), dtype=torch.bool)

        data_mixed_X = torch.cat((data_X_real, data_X_syn), dim=0).to(device)
        data_mixed_Y = torch.cat((data_Y_real, data_Y_syn), dim=0).to(device)
        is_real = torch.cat((is_real_real, is_real_syn), dim=0).to(device)
        trainset_mixed = TensorDataset(data_mixed_X, data_mixed_Y, is_real)


        # coeff_ real and coeff_synth are used to equalize strength of real and synthetic data in the mixed loss regardless of Data Size
        # Requires handling of size 0.
        coeff_real = (num_real_samples + num_synth_samples)/(num_real_samples)
        coeff_synth = (num_real_samples + num_synth_samples)/(num_synth_samples)
        alpha = num_synth_samples/len(trainset) # Adjusting for different dataset sizes, if needed. 
        return trainset_mixed, coeff_real, coeff_synth
    
    
    def fit(self, model, device, ins: FitIns) -> FitRes:
        config = ins.config
        fit_begin = timeit.default_timer()

        # Get training config
        server_round = int(ins.config["server_round"])
        total_rounds = int(ins.config["total_rounds"])
        local_epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])
        learning_rate = float(config["learning_rate"])
        optimizer_str = config["optimizer"]
        criterion_str = config["criterion"]
        optim_kwargs = dict(json.loads(config["optim_kwargs"]))
        perform_evals = config["perform_evals"]

        gen_configs= self.attack_config["GAN_ATTACK_CONFIG"]["HYPER_PARAM"]["GEN_ARGS"]
        num_classes= gen_configs["NUM_CLASSES"]
        latent_size = gen_configs["LATENT_SIZE"]


        attack = np.random.random() < self.attack_config["ATTACK_RATIO"]

        if server_round < self.attack_config["ATTACK_ROUND"] or not attack:
            return super().fit(model=model, device=device, ins=ins)

        # Set model parameters
        model.set_weights(ins.parameters, clone=(not self._process))
        model.to(device)
        # Set Gen model to device for client:
        self.gen_model.to(device)


        mixed_trainset, coeff_real, coeff_synth = self.generate_D_mixed(
            gen_model=self.gen_model,
            attack_config=self.attack_config,
            device=device,
            batch_size=batch_size,
            server_round=server_round,
            trainset=self._trainset,  
        )

        # #switch datasets for the training. From real to mixed (real + synthetic)
        # original_trainset = self._trainset
        # self._trainset = mixed_trainset



        # Stage dataset to GPU
        original_device = self._trainset.data.device
        self._trainset.to_device(device=device)

        # Train model
        trainloader = torch.utils.data.DataLoader(
            self._trainset, batch_size=batch_size, shuffle=True, drop_last=False
        )


        # # Train model
        # trainloader = torch.utils.data.DataLoader(
        #     mixed_trainset, batch_size=batch_size, shuffle=True, drop_last=False
        # )

        criterion = modules.get_criterion(
            criterion_str=criterion_str 
        )
        #, reduction="none"
        optimizer = modules.get_optimizer(
            optimizer_str=optimizer_str,            
            local_model=model,
            learning_rate=learning_rate,
            **optim_kwargs,
        )
        

        num_examples = train_malGAN_discriminator(
            model=model,
            gen_model= self.gen_model,
            trainloader=trainloader, 
            epochs=local_epochs, 
            learning_rate=learning_rate,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_classes=num_classes,
            latent_size=latent_size,
            coeff_real=coeff_real,
            coeff_synth=coeff_synth,
            synth_strength_ratio=self.synth_strength_ratio,
        )

        # num_examples = train_mixed_data(
        #     model=model, 
        #     trainloader=trainloader, 
        #     epochs=local_epochs, 
        #     learning_rate=learning_rate,
        #     criterion=criterion,
        #     optimizer=optimizer,
        #     device=device,
        #     coeff_real=coeff_real,
        #     coeff_synth=coeff_synth,
        #     synth_strength_ratio=self.synth_strength_ratio,
        # )

        # Get weights from the model and stage back to CPU if running as process
        parameters_updated = model.get_weights()
        if self._process: parameters_updated = parameters_updated.cpu()

        fit_duration = timeit.default_timer() - fit_begin

        # Perform necessary evaluations
        ts_loss, ts_accuracy, tr_loss, tr_accuracy = (None, None, None, None)
        if perform_evals:
            ts_loss, ts_accuracy, tr_loss, tr_accuracy = self.perform_evaluations(model, device, trainloader=None, testloader=None)

        # Peforming cleanups
        # del weights, weights_updated, optimizer, trainloader
        del optimizer, trainloader

        # Stage dataset back to CPU
        self._trainset.to_device(device=original_device)


        #Stage client gen model back to original device----------
        self.gen_model.to(original_device)

        # # Restore original dataset
        # self._trainset = original_trainset

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=parameters_updated,
            num_examples=num_examples,
            metrics={
                "client_id": int(self.client_id),
                "fit_duration": fit_duration,
                "train_accu": tr_accuracy,
                "train_loss": tr_loss,
                "test_accu": ts_accuracy,
                "test_loss": ts_loss,
                "attacking": True,
                "client_type": self.client_type,
            },
        )

def train_malGAN_discriminator(
        model: nn.Module,
        gen_model: nn.Module,
        trainloader: torch.utils.data.DataLoader,
        epochs: int,
        device: str,  # pylint: disable=no-member
        learning_rate: float,
        criterion,
        optimizer,
        num_classes: int,
        latent_size: int,
        coeff_real: float,
        coeff_synth: float,
        synth_strength_ratio: float,
    ) -> None:
    

    num_examples = 0
    model.train()
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_real_loss = 0.0
        running_fake_loss = 0.0
        for i, data in enumerate(trainloader):
            images, labels = data[0].to(device), data[1].to(device)

            # Labels
            real_labels = labels
            fake_labels = labels

            # --- Train on real data ---
            optimizer.zero_grad()
            real_outputs = model(images)
            real_loss = -(1 - synth_strength_ratio) * criterion(real_outputs, real_labels)
            real_loss.backward()

            # --- Train on fake data ---
            # gen_model(current_z, current_l)
            input_znoises = torch.randn(labels.size(0), latent_size).to(device)
            
            fake_data = gen_model(input_znoises, fake_labels).detach()  # detach to avoid training generator here
            fake_outputs = model(fake_data)
            fake_loss = synth_strength_ratio * criterion(fake_outputs, fake_labels)
            fake_loss.backward()

            # Update discriminator
            optimizer.step()

            # print statistics
            running_real_loss += real_loss.item()
            running_fake_loss += fake_loss.item()
            num_examples += labels.size(0)

    return num_examples



def train_mixed_data(
        model: nn.Module,
        trainloader: torch.utils.data.DataLoader,
        epochs: int,
        device: str,  # pylint: disable=no-member
        learning_rate: float,
        criterion,
        optimizer,
        coeff_real: float,
        coeff_synth: float,
        synth_strength_ratio: float,
    ) -> None:
    """Helper function to train the model.

    :param model: The local model that needs to be trained.
    :param trainloader: The dataloader of the dataset to use for training.
    :param epochs: Number of training rounds / epochs
    :param device: The device to train the model on i.e. cpu or cuda. 
    :param learning_rate: The initial learning rate the optimizer is using.
    :param criterion: The loss function to use for model training.
    :param optimizer: The optimizer to use for model training.
    :param coeff_real: The coefficient for real data loss.
    :param coeff_synth: The coefficient for synthetic data loss.
    :returns: None.
    """
    # Define loss and optimizer
    # log(
    #     INFO,
    #     f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each"
    # )

    num_examples = 0
    
    # Define loss and optimizer
    log(
        INFO,
        f"Client GAN Attack mixed training: {trainloader.dataset.tensors[0].device}"
    )


    model.train()
    # Train the model
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            images, labels, is_real = data[0].to(device), data[1].to(device), data[2].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(images)
            if criterion.reduction != "none":
                raise ValueError("criterion in Gan_attack must use reduction='none'")
            

            # Mixed loss-------

            is_fake=~is_real
            loss = criterion(outputs, labels)
            weights = torch.zeros_like(loss)
            
            weights[is_real] = -(1 - synth_strength_ratio) * coeff_real
            weights[is_fake] = synth_strength_ratio * coeff_synth
            loss = (weights * loss).mean()
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            num_examples += labels.size(0)

    return num_examples



