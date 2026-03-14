
from abc import ABC, abstractmethod

class Criterion(ABC):
    """Abstract class which allows subclasses to implement criterion sampling."""

    @abstractmethod
    def select(self, client) -> bool:
        """Decide whether a client should be eligible for sampling or not."""

class MaliciousSampling(Criterion):
    """Abstract class which allows subclasses to implement criterion sampling."""

    def select(self, client) -> bool:
        """Decide whether a client should be eligible for sampling or not."""
        if client.client_type != "HONEST":
            return True
