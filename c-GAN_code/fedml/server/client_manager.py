"""ClientManager."""


import math
import random
import threading
from abc import ABC, abstractmethod
from logging import INFO, DEBUG
from typing import Optional

from fedml.common import log
from .criterion import Criterion

class ClientManager(ABC):
    """Abstract base class for managing FedML clients."""

    @abstractmethod
    def num_available(self) -> int:
        """Return the number of available clients.

        Returns
        -------
        num_available : int
            The number of currently available clients.
        """

    @abstractmethod
    def register(self, client) -> bool:
        """Register FedML Client instance.

        Parameters
        ----------
        client : flwr.server.client_proxy.Client

        Returns
        -------
        success : bool
            Indicating if registration was successful. False if Client is
            already registered or can not be registered for any reason.
        """

    @abstractmethod
    def unregister(self, client) -> None:
        """Unregister FedML Client instance.

        This method is idempotent.

        Parameters
        ----------
        client : flwr.server.client_proxy.Client
        """

    @abstractmethod
    def all(self):
        """Return all available clients."""

    @abstractmethod
    def sample(
        self,
        num_clients: int,
        criterion: Optional[Criterion] = None,
        eval_round: bool = False,
    ):
        """Sample a number of FedML Client instances."""


class SimpleClientManager(ClientManager):
    """Provides a pool of available clients."""

    def __init__(self) -> None:
        self.clients = {}

    def __len__(self) -> int:
        """Return the number of available clients.

        Returns
        -------
        num_available : int
            The number of currently available clients.
        """
        return len(self.clients)

    def num_available(self) -> int:
        """Return the number of available clients.

        Returns
        -------
        num_available : int
            The number of currently available clients.
        """
        return len(self)

    def register(self, client) -> bool:
        """Register FedML Client instance.

        Parameters
        ----------
        client : flwr.server.client_proxy.Client

        Returns
        -------
        success : bool
            Indicating if registration was successful. False if Client is
            already registered or can not be registered for any reason.
        """
        if client.cid in self.clients:
            return False

        self.clients[client.cid] = client

        return True

    def unregister(self, client) -> None:
        """Unregister FedML Client instance.

        This method is idempotent.

        Parameters
        ----------
        client : flwr.server.client_proxy.Client
        """
        if client.cid in self.clients:
            del self.clients[client.cid]

    def all(self):
        """Return all available clients."""
        return self.clients

    def sample(
        self,
        num_clients: int,
        criterion: Optional[Criterion] = None,
        eval_round: bool = False,
    ):
        """Sample a number of FedML Client instances."""

        # Sample clients which meet the criterion
        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]

        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []

        sampled_cids = random.sample(available_cids, num_clients)
        return [self.clients[cid] for cid in sampled_cids]

def get_client_manager(user_configs: dict):
    server_configs = user_configs["SERVER_CONFIGS"]
    if server_configs["CLIENTS_MANAGER"] == "SIMPLE":
        return SimpleClientManager()
    else:
        raise ValueError(f"Undefined client manager type {server_configs['CLIENTS_MANAGER']}")