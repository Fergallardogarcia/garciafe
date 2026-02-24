from abc import ABC, abstractmethod
from multiprocessing.connection import Client
from typing import Dict, Optional

import concurrent.futures
from common.typing import Parameters, Scalar, FitIns

from fedml.modules.strategy_functions import get_fit_config_fn
from fedml.server import client_manager
from fedml.strategy import Strategy
from fedml.common.typing import Parameters, Scalar
from fedml.server.client_manager import ClientManager
from fedml.defenses.filters.filter import Filter
from fedml.server.criterion import MaliciousSampling

from models.model import BaseModel
from models.model_loader import load_model 

from modules.trainer import train_generator

class GAN_attack(Filter):
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
            self.executor = None
            if executor_type == "ThreadPool":
                self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
            elif executor_type == "ProcessPool":
                self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers)
            """
        

    
    @property
    def filter_type(self) -> str:
        return "GAN"


    def server_task_fit_round_before(self, global_parameters, server_round: int, executor):
        # Start training of generator from current model parameters
        submitted_fs = {
            executor.submit(self.train_gen, self.parameters, server_round)
        }

        # Wait until malicious generator is trained
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

        #Broadcast gen model to clients
        self.broadcast_gen_model(self.client_manager) #Probably should add security checker for server iteration

        return finished_fs

    def train_gen(self, global_weights: List[torch.Tensor], server_round: int) -> None:
        # Skip first round as we don't have a baseline to perform comparison
        if server_round < self.skip_rounds:
            log(
                INFO,
                "filter_updates: Skipping filteration at server round %s",
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

        # Finished training return
        return self

    
    
    def select_malicious_clients(self, client_manager: ClientManager):
        sampling = MaliciousSampling()
        mal_clients ={client for client in client_manager.clients if sampling.select(client)} 
        return mal_clients

    # Broadcast gen Model (Should only  be done once for now):
    def broadcast_gen_model(self, client_manager: ClientManager):
        mal_clients= self.select_malicious_clients(client_manager)
        #Maybe train gen in here?
        gen_params = self.gen_model.get_weights()
        for client in mal_clients:
            # Attach Gen models or instructions to the selected malicious clients
            client.gen_model.set_weights(gen_params)
        # Configure the attack for the next round of training here 
        return

    # Load GAN models and deliver them with client instructions
    
    def configure_fit_attack(
        self, 
        server_round: int, 
        parameters: Optional[Parameters]=None, 
        client_manager: ClientManager
    ):

        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)
        
        # Return client/config pairs
        # return [(client, self.local_models[id], self.run_devices[id], fit_ins) for id, client in enumerate(clients)]
        return [(client, self.local_models[client.client_id], self.run_devices[client.client_id], fit_ins) for client in clients]




    
    # Get fit configuration for the GAN / extra local models for the malicious clients ============================

    fit_config_fn = get_fit_config_fn(
        total_rounds=user_configs["SERVER_CONFIGS"]["NUM_TRAIN_ROUND"],
        local_epochs=user_configs["CLIENT_CONFIGS"]["LOCAL_EPCH"],
        lr_scheduler=user_configs["CLIENT_CONFIGS"]["LR_SCHEDULER"],
        scheduler_args=user_configs["CLIENT_CONFIGS"]["SCHEDULER_ARGS"],
        local_batchsize=user_configs["CLIENT_CONFIGS"]["BATCH_SIZE"],
        learning_rate=user_configs["CLIENT_CONFIGS"]["LEARN_RATE"],
        initial_lr=user_configs["CLIENT_CONFIGS"]["INITIAL_LR"],
        lr_warmup_steps=user_configs["CLIENT_CONFIGS"]["WARMUP_RDS"],
        optimizer_str=user_configs["CLIENT_CONFIGS"]["OPTIMIZER"],
        criterion_str=user_configs["CLIENT_CONFIGS"]["CRITERION"],
        perform_evals=user_configs["CLIENT_CONFIGS"]["EVALUATE"],
        optim_kwargs=user_configs["CLIENT_CONFIGS"]["OPTIM_ARG"],

    )
    
    evaluate_config_fn = get_evaluate_config_fn(
        total_rounds=user_configs["SERVER_CONFIGS"]["NUM_TRAIN_ROUND"],
        evaluate_bs=user_configs["CLIENT_CONFIGS"]["BATCH_SIZE"],
        criterion_str=user_configs["CLIENT_CONFIGS"]["CRITERION"],
    )
    #============================================================================


    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager
        ):
            """Configure the next round of training."""
            config = {}
            if self.on_fit_config_fn is not None:
                # Custom fit config function provided
                config = self.on_fit_config_fn(server_round)
            fit_ins = FitIns(parameters, config)

            # Sample clients
            sample_size, _ = self.num_fit_clients(
                client_manager.num_available()
            )
            clients = client_manager.sample(
                num_clients=sample_size, criterion=self.fit_criterion, eval_round=False
            )

            # Return client/config pairs
            # return [(client, self.local_models[id], self.run_devices[id], fit_ins) for id, client in enumerate(clients)]
            return [(client, self.local_models[client.client_id], self.run_devices[client.client_id], fit_ins) for client in clients]


    