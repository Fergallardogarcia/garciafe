"""Implementation of Federated Average (FedAvg) strategy."""

from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch

from fedml.server.criterion import MaliciousSampling
from fedml.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    log,
)

from .aggregate import aggregate, weighted_loss_avg
from .strategy import Strategy

class FederatedAverage(Strategy):
    """Federated Averaging strategy.

    Implementation based on https://arxiv.org/abs/1602.05629

    Parameters
    ----------
    fraction_fit : float, optional
        Fraction of clients used during training. In case `min_fit_clients`
        is larger than `fraction_fit * available_clients`, `min_fit_clients`
        will still be sampled. Defaults to 1.0.
    fraction_evaluate : float, optional
        Fraction of clients used during validation. In case `min_evaluate_clients`
        is larger than `fraction_evaluate * available_clients`,
        `min_evaluate_clients` will still be sampled. Defaults to 1.0.
    min_fit_clients : int, optional
        Minimum number of clients used during training. Defaults to 2.
    min_evaluate_clients : int, optional
        Minimum number of clients used during validation. Defaults to 2.
    min_available_clients : int, optional
        Minimum number of total clients in the system. Defaults to 2.
    evaluate_fn : Optional[Callable[[int, Parameters, Dict[str, Scalar]],Optional[Tuple[float, Dict[str, Scalar]]]]]
        Optional function used for validation. Defaults to None.
    on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure training. Defaults to None.
    on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure validation. Defaults to None.
    accept_failures : bool, optional
        Whether or not accept rounds containing failures. Defaults to True.
    initial_parameters : Parameters, optional
        Initial global model parameters.
    fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    inplace : bool (default: True)
        Enable (True) or disable (False) in-place aggregation of model updates.
    """

    def __init__(
        self,
        *,
        local_models: List[any], 
        run_devices: List[str],
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn = None,
        on_fit_config_fn = None,
        on_evaluate_config_fn = None,
        accept_failures: bool = True,
        initial_parameters = None,
        fit_metrics_aggregation_fn = None,
        evaluate_metrics_aggregation_fn = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.initial_parameters = initial_parameters
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.local_models = local_models
        self.run_devices = run_devices
        # if len(self.local_models) != len(self.run_devices) or len(self.local_models) != min_fit_clients:
        #     raise Exception("Number of fit clients must be equal to number of models provided!!!")
        if len(self.local_models) != len(self.run_devices):
            raise Exception("Number of fit clients must be equal to number of models provided!!!")
        self.fit_criterion = None
        self.evaluate_criterion = MaliciousSampling()

    def __repr__(self) -> str:
        return "FederatedAverage"

    def initialize_parameters(
        self, client_manager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ):
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, _ = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, criterion=self.fit_criterion, eval_round=False
        )

        # Return client/config pairs
        # return [(client, self.local_models[id], self.run_devices[id], fit_ins) for id, client in enumerate(clients)]
        return [(client, self.local_models[client.client_id], self.run_devices[client.client_id], fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
        selected: Optional[List[int]] = None,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = []
        for indx, (_, fit_res) in enumerate(results):
            if selected is None:
                weights_results.append((fit_res.parameters, fit_res.num_examples))
            elif indx in selected:
                weights_results.append((fit_res.parameters, fit_res.num_examples))
        parameters_aggregated = aggregate(weights_results) if len(weights_results) > 0 else None

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(
                fit_metrics=fit_metrics, 
                selected=selected, 
                update_norms=self.compute_benign_norms(
                    results=results,
                    aggregated_parameters=parameters_aggregated
                )
            )
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        # Performing cleanups
        del weights_results

        return parameters_aggregated, metrics_aggregated

    def compute_benign_norms(
        self,
        results,
        aggregated_parameters,
    ) -> Optional[Dict[str, Scalar]]:
        """Aggregate fit results for benign clients only using weighted average."""
        return {}           # we no longer need these statistics
        if aggregated_parameters is None:
            return {}
        norm_dict = dict()

        with torch.no_grad():
            # Convert results
            benign_weights = []
            advers_weights = []
            for _, fit_res in results:
                if fit_res.metrics["attacking"]:
                    advers_weights.append((fit_res.parameters, fit_res.num_examples))
                else:
                    benign_weights.append((fit_res.parameters, fit_res.num_examples))

            param_list = [aggregated_parameters]
            
            if len(benign_weights) > 0:
                benign_aggregate = aggregate(benign_weights)
                param_list.append(benign_aggregate)
                flattened_benign_weights = torch.stack([w for (w, _) in benign_weights], dim=0)

            if len(advers_weights) > 0:
                advers_aggregate = aggregate(advers_weights)
                param_list.append(advers_aggregate)
                flattened_malicious_weights = torch.stack([w for (w, _) in advers_weights], dim=0)

            # Compute Norms
            flattened_weights = torch.stack(param_list, dim=0)

            norm_dict["norm_aggregated_update"] = torch.linalg.vector_norm(flattened_weights[0]).item()

            if len(benign_weights) > 0:
                norm_dict["num_benign_updates"] = len(benign_weights)
                norm_dict["norm_benign_updates"] = torch.linalg.vector_norm(flattened_weights[1]).item()
                difference_agg_bgn = flattened_weights[1] - flattened_weights[0]
                norm_dict["norm_diff_agg_benign_aggregated"] = torch.linalg.vector_norm(difference_agg_bgn).item()
                norm_dict["norm_diff_all_benign_aggregated"] = torch.linalg.vector_norm(flattened_benign_weights - flattened_weights[0], dim=1).cpu().detach().numpy()
            else:
                norm_dict["num_benign_updates"] = 0
                norm_dict["norm_benign_updates"] = None
                norm_dict["norm_diff_agg_benign_aggregated"] = None
                norm_dict["norm_diff_all_benign_aggregated"] = None

            if len(advers_weights) > 0:
                norm_dict["num_malicious_updates"] = len(advers_weights)
                norm_dict["norm_malicious_updates"] = torch.linalg.vector_norm(flattened_weights[2]).item()
                difference_agg_bgn = flattened_weights[2] - flattened_weights[0]
                norm_dict["norm_diff_agg_malicious_aggregated"] = torch.linalg.vector_norm(difference_agg_bgn).item()
                norm_dict["norm_diff_all_malicious_aggregated"] = torch.linalg.vector_norm(flattened_malicious_weights - flattened_weights[0], dim=1).cpu().detach().numpy()
            else:
                norm_dict["num_malicious_updates"] = 0
                norm_dict["norm_malicious_updates"] = None
                norm_dict["norm_diff_agg_malicious_aggregated"] = None
                norm_dict["norm_diff_all_malicious_aggregated"] = None

        return norm_dict

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager
    ):
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, _ = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, criterion=self.evaluate_criterion, eval_round=True
        )

        # Return client/config pairs
        # return [(client, evaluate_ins) for client in clients]

        # Return client/config pairs
        return [(client, self.local_models[client.client_id], self.run_devices[client.client_id], evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            # Let's assume we won't perform the global model evaluation on the server side.
            return None
        
        eval_res = self.evaluate_fn(server_round, parameters, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res

        return loss, metrics

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients
