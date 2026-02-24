"""Help function to get requested criterion / loss function."""

import torch.nn as nn

def get_criterion(
        criterion_str: str,    
    ) -> nn.Module:
    """Get requested criterion / loss function.

    :param criterion_str: The name of criterion / loss function to obtain.
    :returns: The requested criterion / loss function.
    """
    assert criterion_str in ["CROSSENTROPY", "NLLL"], f"Invalid criterion {criterion_str} requested."

    if criterion_str == "CROSSENTROPY":
        return nn.CrossEntropyLoss()
    elif criterion_str == "NLLL":
        return nn.NLLLoss()
    else:
        raise ValueError(f"Invalid criterion {criterion_str} requested.")
