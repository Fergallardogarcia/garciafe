from abc import ABC, abstractmethod
from multiprocessing.connection import Client
from typing import Dict, Optional
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



class GAN_attack(Filter):
    


    @property
    def filter_type(self) -> str:
        return "GAN"

    def filter_updates(
        self, 
        client_weights: List[Tuple[Parameters, int]],
        server_round: int
    ) -> Tuple[List[int], Optional[List[Tuple]]]:
        # Perform the GAN based filtering and return the selected client indexes and stats
        selected_indexes = [0]  # Placeholder for selected client indexes
        client_stats = [(0.5, 0.5)]  # Placeholder for client stats (e.g., discriminator scores)
        return selected_indexes, client_stats

    def server_tasks(  
        self,
        global_weights: Parameters,
        server_round: int,
    ):
        # Perform any server side tasks related to GAN training here
        pass
    

    def server_task_fit_round_before(self, server_round: int, executor):
        # Start training of generator from current model parameters
        submitted_fs = {
            executor.submit(self.server_tasks, self.parameters, server_round)
        }
        return submitted_fs



    # Load GAN models and deliver them with client instructions
    def configure_fit_attack(
        self, 
        server_round: int, 
        parameters: Optional[Parameters]=None, 
        client_manager: ClientManager
    ):
        # Configure the attack for the next round of training here (e.g., select clients to attack)
        sampling = MaliciousSampling()
        mal_clients ={client for client in client_manager.clients if sampling.select(client)} 


        for client in mal_clients:
            # Attach GAN models or instructions to the selected malicious clients
            client.gan_model = load_model(attack_config["GAN_ATTACK_CONFIG"]["HYPER_PARAM"]["GEN_ARGS"])  
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)
        
        pass




    
    # Get fit configuration for the GAN / local models extra local models for the malicious clients ============================

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


    