"""A module containing definitions of functions needed to create aggregation strategy."""
from typing import Callable, Dict, Optional, Tuple, List

import torch
from torch.utils.data import Dataset

import json

from fedml.common import (
    Parameters,
    Scalar
)
from fedml.models import load_model
from fedml.modules import evaluate
from .get_lr_scheduler import get_lr_schedule


def get_fit_config_fn(
        total_rounds: int,
        local_epochs: int,
        scheduler_args: Dict,
        local_batchsize: int, 
        learning_rate: float, 
        lr_scheduler: str,
        lr_warmup_steps: int,
        initial_lr: float, 
        optimizer_str: str,
        criterion_str: str,
        perform_evals: bool,
        optim_kwargs: Dict,
    ):

    lr_schedule: List[float] = get_lr_schedule(total_rounds=total_rounds, method=lr_scheduler, warmup_steps=lr_warmup_steps, initial_lr=initial_lr, target_lr=learning_rate, scheduler_args=scheduler_args)

    def fit_config(server_round: int) -> Dict[str, Scalar]:
        """Return a configuration with static batch size and (local) epochs."""

        # Get current learning rate from the schedule
        learning_rate = lr_schedule[server_round-1]

        config: Dict[str, Scalar] = {
            "server_round": str(server_round),
            "total_rounds": str(total_rounds),
            "epochs": str(local_epochs),
            "batch_size": str(local_batchsize),
            "learning_rate": str(learning_rate),
            # "initial_lr": str(initial_lr),
            # "lr_warmup_steps": str(lr_warmup_steps),
            # "lr_scheduler": lr_scheduler,
            # "scheduler_args": json.dumps(scheduler_args),
            "optimizer": optimizer_str,
            "criterion": criterion_str,
            "perform_evals": perform_evals,
            "optim_kwargs": json.dumps(optim_kwargs),
        }
        return config
    return fit_config

def get_evaluate_config_fn(
        total_rounds: int,
        evaluate_bs: int, 
        criterion_str: str,
    ):
    def fit_config(server_round: int) -> Dict[str, Scalar]:
        """Return a configuration with static batch size and (local) epochs."""
        config: Dict[str, Scalar] = {
            "server_round": str(server_round),
            "total_rounds": str(total_rounds),
            "batch_size": str(evaluate_bs),
            "criterion": criterion_str,
        }
        return config
    return fit_config

def get_evaluate_fn(
    testset: Dataset,
    model_configs: dict,
    device: str,
) -> Callable[[int, Parameters, Dict[str, Scalar]], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""
    model = load_model(model_configs=model_configs)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False)

    def evaluate_fn(
            server_round: int, 
            weights: Parameters,
            config: Dict[str, Scalar],
        ) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""
        model.set_weights(weights)
        model.to(device)
        loss, accuracy, _ = evaluate(model, testloader, device=device)

        return loss, {"accuracy": accuracy}

    return evaluate_fn
