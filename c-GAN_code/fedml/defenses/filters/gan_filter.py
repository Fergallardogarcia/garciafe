from typing import Dict, List, Tuple, Optional
from logging import INFO, DEBUG
from collections import defaultdict

import concurrent.futures

import torch
import torch.nn as nn

import os
import threading

from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import scipy.stats
from sklearn.cluster import KMeans

from fedml.common import Parameters, log
from fedml.server.client_manager import ClientManager
from fedml.models import load_model
from fedml.modules import get_optimizer, get_criterion, train_generator, evaluate_gan

from .filter import Filter

class GenerativeFilter(Filter):
    def __init__(
            self, 
            gen_configs, 
            dis_configs, 
            train_configs, 
            filter_configs, 
            cuda_device: Optional[str] = "cuda",
            skip_rounds = 1,
        ) -> None:
        self.gen_configs = gen_configs
        self.dis_configs = dis_configs
        self.train_configs = train_configs
        self.filter_configs = filter_configs
        

        which_device = train_configs["DEVICE"]
        if which_device in ["auto", "cuda"] and torch.cuda.is_available():
            self.device = cuda_device
        else:
            self.device = "cpu"

        # Setup models (both generator and discriminator)
        self.gen_model = load_model(model_configs=gen_configs)
        if self.gen_configs["SET_WEIGHTS"]:
            # self.gen_model.load_state_dict(torch.load(gen_configs["WEIGHT_PATH"], weights_only=False))
            init_gen_weights = torch.load(gen_configs["WEIGHT_PATH"])
            self.gen_model.set_weights(init_gen_weights)
        self.dis_model = load_model(model_configs=dis_configs)


        self.criterion = get_criterion(criterion_str=train_configs["CRITERION"])


        
        # torch.get_default_device()
        
        # Setup random data needed for dataset generation
        
        # Use trained generator to create samples
        self.num_classes = gen_configs["NUM_CLASSES"]
        self.samples_per_class = filter_configs["SAMPLES_PER_CLASS"]
        self.latent_size = gen_configs["LATENT_SIZE"]
        self.skip_rounds = skip_rounds

        # Generate input randomness
        self.input_znoises = torch.randn(self.num_classes * self.samples_per_class, self.latent_size)
        self.input_classes = torch.arange(self.num_classes, dtype=torch.int64).repeat_interleave(self.samples_per_class)

    @property
    def filter_type(self):
        """Returns current filter's type."""
        return "GAN"

    # Wraper methods for server hooks-------------------------------
    def server_fit_round_before(
                self, 
                global_parameters, 
                server_round: int, 
                executor,
                client_instructions = None,
    ):
        # Start training of generator from current model parameters
        submitted_fs ={
            executor.submit(self.train_gen, global_parameters, server_round)
        }
       
        return submitted_fs
        
        

    def server_fit_round_after(self, gen_weights):
        if gen_weights is None:
            return None
        
        self.gen_model.set_weights(gen_weights)
        gen_dataset= generate_dataset(
            gen_model=self.gen_model,
            input_znoises=self.input_znoises,
            input_classes=self.input_classes,
            device=self.device,
            batch_size=1024,
        )

        return self.filter_round, self.log_filter_stats, gen_dataset

    
    
    
    #  Server tasks------------------------------
    def train_gen(self, global_weights: List[torch.Tensor], server_round: int) -> None:
        
        # Skip first round as we don't have a baseline to perform comparison
        if server_round < self.skip_rounds:
            log(
                INFO,
                "filter_updates: Skipping filteration at server round %s",
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

        
        # Return trained generator weights
        return self.gen_model.get_weights()



    def filter_round(self, server_round: int, results):
        (selected_indexes, client_stats, gen_dataset) = self.filter_updates(
            client_weights=[(fit_res.parameters, fit_res.num_examples) for _, fit_res in results],
            client_info=[(fit_res.metrics["client_id"], fit_res.metrics["attacking"]) for _, fit_res in results],
            server_round=server_round,
        )
        return selected_indexes, client_stats, gen_dataset



    def log_filter_stats(self, metrics_aggregated, client_stats, results):
        # Logging filtering stats of clients
        if self.filter_type == "GAN" and client_stats is not None:
            metrics_aggregated["filter_loss"] = dict()
            metrics_aggregated["filter_accu_all"] = dict()
            metrics_aggregated["filter_accu_cls"] = dict()
        
            # Compile / collect results
            client_ids = [res.metrics["client_id"] for _, res in results]
            for index, cid in enumerate(client_ids):
                metrics_aggregated["filter_loss"][f"client_{cid}"] = client_stats["avg_loss"][index]
                metrics_aggregated["filter_accu_all"][f"client_{cid}"] = client_stats["accu_all"][index]
                metrics_aggregated["filter_accu_cls"][f"client_{cid}"] = client_stats["accu_cls"][index]

        if self.filter_type == "KRUM":
            metrics_aggregated["distances"] = dict()

            # Compile / collect results
            client_ids = [res.metrics["client_id"] for _, res in results]
            for index, cid in enumerate(client_ids):
                metrics_aggregated["distances"][f"client_{cid}"] = client_stats["distances"][index, :]
        return metrics_aggregated



    # Functionalities related to filteration process------------------------------
    def filter_updates(
            self, 
            client_weights: List[Tuple[Parameters, int]],
            client_info: List[Tuple[str, bool]],
            server_round: int
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
            return [indx for indx in range(len(client_weights))], None, None

        # Generate a new validation dataset
        gen_dataset = generate_dataset(
            gen_model=self.gen_model,
            input_znoises=self.input_znoises,
            input_classes=self.input_classes,
            device=self.device,
            batch_size=1024,
        )

        # Perform filteration 
        select_cids, client_stats, attack_labels = perform_filteration(
            filter_configs=self.filter_configs,
            dis_model=self.dis_model,
            gen_dataset=gen_dataset,
            client_weights=client_weights,
            client_info=client_info,
            criterion=self.criterion,
            num_classes=self.num_classes,
            samples_per_class=self.samples_per_class,
            device=self.device
        )

        acc_reg_labels = ["+" if idx in select_cids else "~" for idx, _ in enumerate(client_stats["accu_all"])]
        hon_mal_labels = ["A" if att else "h" for idx, att in enumerate(attack_labels)]
        combined= [f"{a}{b}" for a, b in zip(acc_reg_labels, hon_mal_labels)]
        acc_labeled = [f"{prefix}{acc:0.4f}" for prefix, acc in zip(acc_reg_labels, client_stats["accu_all"])]
        loss_labeled = [f"{prefix}{loss:0.4f}" for prefix, loss in zip(hon_mal_labels, client_stats["avg_loss"])]
        acc_pairs = list(zip(client_stats["accu_all"], combined))
        loss_pairs = list(zip(client_stats["avg_loss"], combined))

        log(
            INFO,
            "filter_updates: (%s, total: %s, selected: %s)",
            server_round,
            len(client_weights),
            len(select_cids),
        )

        log(
            DEBUG,
            "filter_updates: accuracies %s,",
            acc_pairs,
            
        )

        log(
            DEBUG,
            "filter_updates: average loss %s",
            loss_pairs,
            
        )


        # Return selected client ids and stats
        return select_cids, client_stats, gen_dataset

def perform_filteration(
        filter_configs, 
        dis_model, 
        gen_dataset, 
        client_weights, 
        client_info,
        criterion,
        num_classes, 
        samples_per_class, 
        device
    ):
    # Create a dataloader of the generator data
    dataloader = DataLoader(gen_dataset, batch_size=2048, shuffle=False) #From 2048 to 1024
    threshold = filter_configs["BASELINE_OVERALL_MIN_ACC"]

    # Perform evaluation of all clients
    select_ids = []
    client_stats = defaultdict(list)
    for index, (weights, _) in enumerate(client_weights):
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
        client_stats["avg_loss"].append(stats[0])
        client_stats["accu_all"].append(stats[1])
        client_stats["accu_cls"].append(stats[2])

    # Perform the actual selection based on criteria
    if filter_configs["FILTERATION_TYPE"] == "BASELINE-OVERALL":
        for index, client_accu in enumerate(client_stats["accu_all"]):
            # Perform selection check for current client
            if client_accu >= threshold:
                select_ids.append(index)

    elif filter_configs["FILTERATION_TYPE"] == "MEAN-LOSS":
        # Compure average loss and make selections
        avg_loss = sum(client_stats["avg_loss"]) / len(client_stats["avg_loss"])
        for index, client_loss in enumerate(client_stats["avg_loss"]):
            if client_loss <= avg_loss: select_ids.append(index)

    elif filter_configs["FILTERATION_TYPE"] == "MEDIAN-LOSS":
        # Compure average loss and make selections
        med_loss = np.median(client_stats["avg_loss"])
        for index, client_loss in enumerate(client_stats["avg_loss"]):
            if client_loss <= med_loss: select_ids.append(index)

    elif filter_configs["FILTERATION_TYPE"] == "MEDIAN-ACCURACY":
        # Compure average loss and make selections
        med_accu = np.median(client_stats["accu_all"])
        for index, client_accu in enumerate(client_stats["accu_all"]):
            if client_accu >= med_accu: select_ids.append(index)

    elif filter_configs["FILTERATION_TYPE"] == "MIXED-LOSS":
        # Compure average loss and make selections
        med_loss = np.median(client_stats["avg_loss"])
        avg_loss = np.mean(client_stats["avg_loss"])
        baseline_loss = med_loss if med_loss < avg_loss else avg_loss

        for index, client_loss in enumerate(client_stats["avg_loss"]):
            if client_loss <= baseline_loss: select_ids.append(index)

    elif filter_configs["FILTERATION_TYPE"] == "MIXED-ACCURACY":
        # Compure average loss and make selections
        med_accu = np.median(client_stats["accu_all"])
        avg_accu = np.mean(client_stats["accu_all"])
        baseline_accu = med_accu if med_accu > avg_accu else avg_accu
        for index, client_accu in enumerate(client_stats["accu_all"]):
            if client_accu >= baseline_accu: select_ids.append(index)

    elif filter_configs["FILTERATION_TYPE"] == "MIXED-2-LOSS":
        # Compure average loss and make selections
        baseline_loss =  0.5 * (np.median(client_stats["avg_loss"]) + np.mean(client_stats["avg_loss"]))

        for index, client_loss in enumerate(client_stats["avg_loss"]):
            if client_loss <= baseline_loss: select_ids.append(index)

    elif filter_configs["FILTERATION_TYPE"] == "MIXED-2-ACCURACY":
        # Compure average loss and make selections
        baseline_accu = 0.5 * (np.median(client_stats["accu_all"]) + np.mean(client_stats["accu_all"]))
        for index, client_accu in enumerate(client_stats["accu_all"]):
            if client_accu >= baseline_accu: select_ids.append(index)

    elif filter_configs["FILTERATION_TYPE"] == "CLUSTER-ACCURACY":
        # Perform clustering and evaluation
        client_accuracy = np.array(client_stats["accu_all"]).reshape(-1, 1)
        kmodel = KMeans(n_clusters=2, n_init=20)
        kmodel.fit(client_accuracy)
        client_clusters = kmodel.predict(client_accuracy)
        select_ids = np.where(client_clusters == client_clusters[client_accuracy.argmax()])[0].tolist()

    elif filter_configs["FILTERATION_TYPE"] == "CLUSTER-LOSS":
        # Perform clustering and evaluation
        client_losses = np.array(client_stats["avg_loss"]).reshape(-1, 1)
        kmodel = KMeans(n_clusters=2, n_init=20)
        kmodel.fit(client_losses)
        client_clusters = kmodel.predict(client_losses)
        select_ids = np.where(client_clusters == client_clusters[client_losses.argmin()])[0].tolist()

    else:
        raise ValueError(f"Invalid filteration type {filter_configs['FILTERATION_TYPE']} specified.")
    
    
    attack_labels = [cinf[1] for cinf in client_info]

    return (select_ids, client_stats, attack_labels)


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


