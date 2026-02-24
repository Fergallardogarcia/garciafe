"""Server strategy."""


from abc import ABC, abstractmethod
from multiprocessing.connection import Client
from typing import Optional


from fedml.server import client_manager
from fedml.strategy import Strategy
from fedml.common.typing import Parameters, Scalar
from fedml.server.client_manager import ClientManager
from fedml.defenses.filters.filter import Filter



class Plan(ABC):
    """Abstract base class for server strategy implementations."""


    def __init__(self,
                 client_manager: Optional[ClientManager] = None,
                 agg_strat: Optional[Strategy] = None,
                 filter_strat: Optional[Filter] = None,
    ):
        self.client_manager = client_manager
        self.agg_strat = agg_strat 
        self.filter_strat = filter_strat


    @abstractmethod
    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize the (global) model parameters.
        """

    @abstractmethod
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ):
        """Configure the next round of training.
        """


    def distribute_model(self, parameters: Parameters):
        """Distribute the current model parameters to all clients."""
        self.client_manager.broadcast_parameters(parameters)

    @abstractmethod
    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate training results.
        """

    @abstractmethod
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ):
        """Configure the next round of evaluation.
        """

    @abstractmethod
    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ):
        """Aggregate evaluation results.
        """

    @abstractmethod
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        """Evaluate the current model parameters.
        """