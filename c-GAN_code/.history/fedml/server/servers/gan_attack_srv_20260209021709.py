"""Implementation of a Normal Server extending the built in FedML Server Class."""

# import multiprocessing
# import multiprocessing.pool
# import torch.multiprocessing as mp
import concurrent.futures
from logging import INFO, WARNING, DEBUG

from typing import Dict, Optional

from fedml.common import (
    Parameters,
    Scalar,
    log
)
from fedml.defenses import create_filter

from .server import Server, fit_clients

from mo
class ServerMaliciousGenerator(Server):
    def __init__(
        self, 
        *, 
        client_manager, 
        experiment_manager = None,
        strategy = None,
        user_configs: Optional[Dict] = None,
        initial_parameters = None,
        executor_type: str = "ThreadPool",
        filter = None,
        
    ) -> None:        
        super().__init__(
            client_manager=client_manager, 
            experiment_manager=experiment_manager, 
            strategy=strategy, 
            user_configs=user_configs, 
            initial_parameters=initial_parameters,
            executor_type=executor_type,
            filter=filter,
        )
    self.malGAN= 
    def srv_tsk(self, executor, user_configs):
        super().srv_tsk(executor, user_configs)

        return submitted_fs

    
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


