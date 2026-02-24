"""Server strategy."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

import concurrent

from fedml.common.typing import Parameters

class Filter(ABC):
    @property
    def filter_type(self) -> str:
        """Filter type. Used to distinguish between different types of filters
        """
        return "BASE"

    # @abstractmethod
    # def filter_updates(
    #     self, 
    #     client_weights: List[Tuple[Parameters, int]],
    #     server_round: int
    # ) -> Tuple[List[int], Optional[List[Tuple]]]:
    #     """Filter updates given global weights and client updates.
    #     """

    # @abstractmethod
    # def server_tasks(
    #     self,
    #     global_weights: Parameters,
    #     server_round: int,
    # ):
    #     """Perform any server side tasks that can run in parallel to client 
    #     training. Useful to perform GAN training in parallel to client training.
    #     """

    
    def server_fit_round_before(
              self, 
              global_parameters, 
              server_round: int, 
              executor,
              client_manager,
    ):
        return set()
    
    def server_fit_round_after(self):
        return 