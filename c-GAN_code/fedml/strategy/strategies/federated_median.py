"""Implementation of Federated Average (FedAvg) strategy."""

from logging import WARNING, DEBUG
from typing import Callable, Dict, List, Optional, Tuple, Union

from fedml.common import (
    FitRes,
    Parameters,
    Scalar,
    log
)
from .federated_average import FederatedAverage
from .aggregate import aggregate_median


class FederatedMedian(FederatedAverage):

    def __repr__(self) -> str:
        return "FederatedMedian"

    def aggregate_fit(
        self,
        server_round: int,
        results: FitRes,
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
        # weights_results = [(fit_res.parameters, fit_res.num_examples) for _, fit_res in results]

        parameters_aggregated = aggregate_median(weights_results)

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

        return parameters_aggregated, metrics_aggregated
