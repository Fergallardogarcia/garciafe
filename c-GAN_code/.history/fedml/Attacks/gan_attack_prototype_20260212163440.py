from abc import ABC, abstractmethod
from multiprocessing.connection import Client
from typing import Optional


from fedml.server import client_manager
from fedml.strategy import Strategy
from fedml.common.typing import Parameters, Scalar
from fedml.server.client_manager import ClientManager
from fedml.defenses.filters.filter import Filter
from fedml.server.criterion import MaliciousSampling
from fedml.attacks.gan_attack_prototype import GAN_attack



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

    def configure_fit_attack(
        self, 
        server_round: int, 
        parameters: Parameters, 
        client_manager: ClientManager
    ):
        # Configure the attack for the next round of training here (e.g., select clients to attack)
        sampling = MaliciousSampling()
        mal_clients ={client for client in client_manager.clients if sampling.select(client)} 
            
        pass










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


    def server_task_fit_round_before(self, server_round: int, executor):
        # Start training of generator from current model parameters
        submitted_fs = {
            executor.submit(self.server_tasks, self.parameters, server_round)
        }
        return submitted_fs