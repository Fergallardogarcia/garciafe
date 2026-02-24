from abc import ABC, abstractmethod
from multiprocessing.connection import Client
from typing import Optional


from fedml.server import client_manager
from fedml.strategy import Strategy
from fedml.common.typing import Parameters, Scalar
from fedml.server.client_manager import ClientManager
from fedml.defenses.filters.filter import Filter
from fedml.server.criterion import MaliciousSampling
from attacks.Gan_prototype import GAN_prototype



class GAN_prototype(Filter):
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
        sampling= MaliciousSampling()
        for client in client_manager.clients:
            sampling.select(client)
        pass