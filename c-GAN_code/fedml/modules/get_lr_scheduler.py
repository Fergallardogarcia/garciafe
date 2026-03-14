"""Help function to get requested learning rate scheduler."""

from typing import Optional, Dict

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR, SequentialLR, ExponentialLR

def get_lr_scheduler(
    optimiser: Optimizer,
    total_epochs: int,
    method: Optional[str] = "STATIC",
    warmup_steps: int = 0,
    initial_lr: Optional[float] = None,
    target_lr: Optional[float] = None,
    kwargs: Optional[Dict] = None,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Get the learning rate scheduler.

    :param optimiser: The optimiser for which to get the scheduler.
    :param total_epochs: The total number of epochs.
    :param method: The method to use for the scheduler.
    :returns: The learning rate scheduler.
    """
    schedulers_list = []
    milestones = []
    if warmup_steps > 0:
        factor = (target_lr / initial_lr) ** (1/warmup_steps)
        schedulers_list.append(ExponentialLR(optimizer=optimiser, gamma=factor))
        milestones.append(warmup_steps)

    if method == "STATIC":
        schedulers_list.append(MultiStepLR(optimiser, [total_epochs + 1]))
    elif method == "3-STEP":
        schedulers_list.append(
            MultiStepLR(
                optimiser, [int(0.25 * total_epochs), int(0.5 * total_epochs), int(0.75 * total_epochs)], gamma=0.1
            )
        )
    elif method == "CUSTOM":
        schedulers_list.append(
            MultiStepLR(optimiser, **kwargs)
        )
    else:
        raise ValueError(f"{method} scheduler not currently supported.")
    
    return SequentialLR(optimizer=optimiser, schedulers=schedulers_list, milestones=milestones)


def get_lr_schedule(
    total_rounds: int,
    method: Optional[str] = "STATIC",
    warmup_steps: int = 0,
    initial_lr: Optional[float] = None,
    target_lr: Optional[float] = None,
    scheduler_args: Optional[Dict] = None,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Get the learning rate scheduler.

    :param optimiser: The optimiser for which to get the scheduler.
    :param total_epochs: The total number of epochs.
    :param method: The method to use for the scheduler.
    :returns: The learning rate scheduler.
    """
    learn_rate_schedule = []
    if warmup_steps > 0:
        factor = (target_lr / initial_lr) ** (1/warmup_steps)
        learn_rate_schedule.extend(
            [initial_lr*(factor**step) for step in range(warmup_steps)]
        )

    if method == "STATIC":
        learn_rate_schedule.extend([target_lr for _ in range(total_rounds-warmup_steps)])
    elif method == "3-STEP":
        current_lr = target_lr
        decay_steps = [int(0.25 * total_rounds), int(0.50 * total_rounds), int(0.75 * total_rounds)]
        for current_round in range(total_rounds):
            if current_round in decay_steps: 
                current_lr *= scheduler_args["gamma"]
            if current_round >= warmup_steps:
                learn_rate_schedule.append(current_lr)
    elif method == "CUSTOM":
        decay_steps = [int(milestone * total_rounds) for milestone in scheduler_args["milestones"]]
        current_lr = target_lr
        for current_round in range(total_rounds):
            if current_round in decay_steps: 
                current_lr *= scheduler_args["gamma"]
            if current_round >= warmup_steps:
                learn_rate_schedule.append(current_lr)

    return learn_rate_schedule
