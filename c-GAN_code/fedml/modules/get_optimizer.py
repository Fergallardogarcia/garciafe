"""Help function to get requested optimizer."""

from typing import Dict

from torch.optim import Optimizer, SGD, Adam
import torch.nn as nn

def get_optimizer(
        optimizer_str: str,
        local_model: nn.Module,
        learning_rate: float,
        **kwargs: Dict,
    ) -> Optimizer:
    """Get requested optimizer function.

    :param optimizer_str: The name of optimiser to obtain.
    :param local_model: The local model for which to obtain an optimizer.
    :param learning_rate: The initial learning rate the optimizer should use.
    :param kwargs: Any additional kwargs for the optimizer.
    :returns: The requested optimizer function.
    """
    assert optimizer_str in ["SGD", "ADAM"], f"Invalid optimizer {optimizer_str} requested."

    if optimizer_str == "SGD":
        return SGD(local_model.parameters(), lr=learning_rate, **kwargs)
    elif optimizer_str == "ADAM":
        return Adam(local_model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Invalid criterion {optimizer_str} requested.")
