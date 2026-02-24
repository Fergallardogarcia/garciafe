
from typing import List, Tuple, Optional

import torch

# from fedml.strategy.strategies.aggregate import _flatten_weights
from fedml.common import Parameters

from .filter import Filter

class KrumFilter(Filter):
    def __init__(self, num_malicious_clients, num_clients_to_keep, **kwargs) -> None:
        self.num_malicious_clients = num_malicious_clients
        self.num_clients_to_keep = num_clients_to_keep

    @property
    def filter_type(self):
        """Returns current filter's type."""
        return "KRUM"

    def server_tasks(self, global_weights: Parameters, server_round: int):
        return

    def filter_updates(
            self, 
            client_weights: List[Tuple[Parameters, int]], 
            server_round: int
        ) -> Tuple[List[int], Optional[List[Tuple]]]:
        """Function to select updates based on (multi) krum filtering"""

        # Create a list of weights and ignore the number of examples
        weights_list = [weights for weights, _ in client_weights]

        # Compute distances between vectors
        distance_matrix = _compute_distances(weights_list)

        # For each client, take the n-f-2 closest parameters vectors
        num_closest = max(1, len(weights_list) - self.num_malicious_clients - 2)
        sorted_indices = torch.argsort(distance_matrix, dim=1)

        # Compute the score for each client, that is the sum of the distances
        # of the n-f-2 closest parameters vectors
        scores = torch.sum(
            distance_matrix.gather(1, sorted_indices[:, 1 : (num_closest + 1)]), dim=1
        )

        if self.num_clients_to_keep > 0:
            # Choose to_keep clients and return their average (MultiKrum)
            best_indices = torch.argsort(scores, descending=False)[:self.num_clients_to_keep]  # noqa: E203
        else:
            best_indices = [torch.argmin(scores).item()]
            # best_results = [results[i] for i in best_indices]
            # return aggregate(best_results)
        
        stats = {
            "distances": distance_matrix
        }

        # Return the model parameters that minimize the score (Krum)
        return best_indices, stats

def _compute_distances(weights: list[Parameters]) -> torch.Tensor:
    """Compute distances between vectors.

    Input: weights - list of weights vectors
    Output: distances - matrix distance_matrix of squared distances between the vectors
    """
    flat_w = torch.stack(weights, dim=0)

    distance_matrix = torch.zeros((len(weights), len(weights)))
    for i, flat_w_i in enumerate(flat_w):
        for j, flat_w_j in enumerate(flat_w):
            delta = flat_w_i - flat_w_j
            norm = torch.linalg.norm(delta)
            distance_matrix[i, j] = norm**2
    return distance_matrix
