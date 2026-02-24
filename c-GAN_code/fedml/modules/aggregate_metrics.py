"""A function to aggregate fit metrics."""

import torch
from typing import List, Tuple, Dict
from fedml.common import Metrics

def aggregate_fit_metrics(
        fit_metrics: List[Tuple[int, Metrics]],
        selected: List[bool] = None,
        weight_pi: Dict[int, float] = None,
        update_norms: Dict[str, float] = None,
        # wandb_logging: bool = False
    ) -> Metrics:

    metrics_aggregated = {
        "sampled": list(),
        "train_accu": dict(),
        "train_loss": dict(),
        "test_accu": dict(),
        "test_loss": dict(),
        "attacking": dict(),
        "client_type": dict(),
        "fit_duration": dict(),
        "num_examples": dict(),
    }

    # Add metric selected conditionally
    if selected is not None:
        metrics_aggregated["selected"] = dict()
    
    # Add metric weight_pi conditionally
    if weight_pi is not None:
        metrics_aggregated["weight_pi"] = dict()
        if torch.is_tensor(weight_pi):
            weight_pi = weight_pi.cpu().detach().numpy()
    
    # Add metric norms conditionally
    if update_norms is not None:
        for key in update_norms.keys():
            metrics_aggregated[key] = update_norms[key]
        # if "benign_upd_norm" in update_norms.keys(): metrics_aggregated["benign_upd_norm"] = update_norms["benign_upd_norm"]
        # if "aggregated_norm" in update_norms.keys(): metrics_aggregated["aggregated_norm"] = update_norms["aggregated_norm"]
        # if "difference_norm" in update_norms.keys(): metrics_aggregated["difference_norm"] = update_norms["difference_norm"]

    # Run through all experiments and accumulate
    # desired results in pre-defined format
    for indx, (num_examples, client_dict) in enumerate(fit_metrics):
        metrics_aggregated["sampled"].append(client_dict['client_id'])
        metrics_aggregated["train_accu"][f"client_{client_dict['client_id']}"] = client_dict["train_accu"]
        metrics_aggregated["train_loss"][f"client_{client_dict['client_id']}"] = client_dict["train_loss"]
        metrics_aggregated["test_accu"][f"client_{client_dict['client_id']}"] = client_dict["test_accu"]
        metrics_aggregated["test_loss"][f"client_{client_dict['client_id']}"] = client_dict["test_loss"]
        metrics_aggregated["attacking"][f"client_{client_dict['client_id']}"] = client_dict["attacking"]
        metrics_aggregated["client_type"][f"client_{client_dict['client_id']}"] = client_dict["client_type"]
        metrics_aggregated["fit_duration"][f"client_{client_dict['client_id']}"] = client_dict["fit_duration"]
        metrics_aggregated["num_examples"][f"client_{client_dict['client_id']}"] = num_examples
        if selected is not None:
            metrics_aggregated["selected"][f"client_{client_dict['client_id']}"] = indx in selected
        if weight_pi is not None:
            metrics_aggregated["weight_pi"][f"client_{client_dict['client_id']}"] = weight_pi[indx]

    return metrics_aggregated


def aggregate_evaluate_metrics(
        eval_metrics: List[Tuple[int, Metrics]],
    ) -> Metrics:

    metrics_aggregated = {
        "train_samples": list(),
        "train_success": list(),
        "train_asr": list(),
        "test_samples": list(),
        "test_success": list(),
        "test_asr": list(),
    }

    # Run through all experiments and accumulate
    # desired results in pre-defined format
    for indx, (num_examples, client_dict) in enumerate(eval_metrics):
        if "train_samples" in client_dict.keys():
            metrics_aggregated["train_samples"].append(client_dict["train_samples"])
            metrics_aggregated["train_success"].append(client_dict["train_success"])
            metrics_aggregated["train_asr"].append(client_dict["train_asr"])
            metrics_aggregated["test_samples"].append(client_dict["test_samples"])
            metrics_aggregated["test_success"].append(client_dict["test_success"])
            metrics_aggregated["test_asr"].append(client_dict["test_asr"])

    return metrics_aggregated