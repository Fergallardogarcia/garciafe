"""Aggregation functions for strategy implementations."""

import math
from functools import reduce
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch

from fedml.common import Parameters

def aggregate(results: list[tuple[Parameters, int]]) -> Parameters:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum(num_examples for (_, num_examples) in results)

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [weights * num_examples for weights, num_examples in results]

    # Compute average weights of each layer
    weights_prime = reduce(torch.add, weighted_weights) / num_examples_total

    return weights_prime

def weighted_loss_avg(results: list[tuple[int, float]]) -> float:
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum(num_examples for (num_examples, _) in results)
    weighted_losses = [num_examples * loss for num_examples, loss in results]
    return sum(weighted_losses) / num_total_evaluation_examples


def aggregate_median(results: list[tuple[Parameters, int]]) -> Parameters:
    """Compute median."""
    # Create a list of weights and ignore the number of examples
    weights = [weights for weights, _ in results]

    # Compute median weight of each layer
    median_w = (
        torch.stack(weights, dim=0)
        .float()
        .quantile(q=0.5, dim=0, interpolation="midpoint")
    )

    return median_w

def aggregate_geometric_median(results: list[tuple[Parameters, int]]) -> Parameters:
    """Compute median."""
    # Create a list of weights and ignore the number of examples
    models = [models for (models, _) in results]
    alphas = torch.tensor(
        [num_examples for (_, num_examples) in results],
        dtype=torch.float32,
        device=models[0][0].device,
    )
    alphas /= alphas.sum()

    geomedian, geo_weights = _compute_geometric_median(
        flat_models=torch.stack(models, dim=0), alphas=alphas
    )
    geo_weights /= geo_weights.max()

    return geomedian, geo_weights


def aggregate_krum(
    results: list[tuple[Parameters, int]], num_malicious: int, to_keep: int
) -> Parameters:
    """Compute krum or multi-krum."""
    # Create a list of weights and ignore the number of examples
    weights = [weights for weights, _ in results]

    # Compute distances between vectors
    distance_matrix = _compute_distances(weights)

    # For each client, take the n-f-2 closest parameters vectors
    num_closest = max(1, len(weights) - num_malicious - 2)
    sorted_indices = torch.argsort(distance_matrix, dim=1)

    # Compute the score for each client, that is the sum of the distances
    # of the n-f-2 closest parameters vectors
    scores = torch.sum(
        distance_matrix.gather(1, sorted_indices[:, 1 : (num_closest + 1)]), dim=1
    )

    if to_keep > 0:
        # Choose to_keep clients and return their average (MultiKrum)
        best_indices = torch.argsort(scores, descending=False)[:to_keep]  # noqa: E203
        best_results = [results[i] for i in best_indices]
        return aggregate(best_results)

    # Return the model parameters that minimize the score (Krum)
    return weights[torch.argmin(scores).item()]


def aggregate_trimmed_average(
    results: list[tuple[Parameters, int]], proportiontocut: float
) -> Parameters:
    """Compute trimmed average."""
    # Create a list of weights and ignore the number of examples
    weights = [weights for weights, _ in results]
    trimmed_w: Parameters = _trim_mean(
        torch.stack(weights, dim=0).float(), proportiontocut=proportiontocut
    )
    return trimmed_w

def _trim_mean(array: torch.Tensor, proportiontocut: float) -> torch.Tensor:
    """Compute trimmed mean along axis=0."""
    axis = 0
    nobs = array.size(axis)
    todrop = int(proportiontocut * nobs)
    result = torch.mean(
        torch.topk(
            torch.topk(array, k=nobs - todrop, dim=0, largest=True)[0],
            k=nobs - (2 * todrop),
            dim=0,
            largest=False,
        )[0],
        dim=0,
    )

    return result


def _compute_geometric_median(
    flat_models: torch.Tensor, alphas: torch.Tensor, maxiter=100, tol=1e-20, eps=1e-8
) -> Tuple[Parameters, torch.Tensor]:
    """Compute the geometric median weights using Weiszfeld algorithm.

    :param models: An list of model weights
    :param maxiter: Maximum number of iterations to run
    :param tol: Tolerance threshold
    :param eps: Minimum threshold (to avoid division by zero)
    :returns: An array of geometric median weights
    """

    # Compute geometric median using Weiszfeld algorithm
    with torch.no_grad():
        # Find initial estimate using the initial guess
        geomedian = alphas @ flat_models / alphas.sum()

        for _ in range(maxiter):
            prev_geomedian = geomedian  # .detach().clone()
            dis = torch.linalg.vector_norm(flat_models - geomedian, dim=1)
            weights = alphas / torch.clamp(dis, min=eps)
            geomedian = weights @ flat_models / weights.sum()

            if torch.linalg.norm(prev_geomedian - geomedian) <= tol * torch.linalg.norm(
                geomedian
            ):
                break

    return geomedian, weights

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
