"""Implementation of a Normal Server extending the built in FedML Server Class."""

# import multiprocessing
# import multiprocessing.pool
# import torch.multiprocessing as mp
import concurrent.futures
from logging import INFO, WARNING, DEBUG

from typing import Dict, Optional

from fedml.common import (
    Parameters,
    Scalar,
    log
)
from fedml.defenses import create_filter

from .server import Server, fit_clients

class FilterServer(Server):
    def __init__(
        self, 
        *, 
        client_manager, 
        experiment_manager = None,
        strategy = None,
        user_configs: Optional[Dict] = None,
        initial_parameters = None,
        executor_type: str = "ThreadPool",
        
    ) -> None:        
        super().__init__(
            client_manager=client_manager, 
            experiment_manager=experiment_manager, 
            strategy=strategy, 
            user_configs=user_configs, 
            initial_parameters=initial_parameters,
            executor_type=executor_type,
        )
        self.filter = create_filter(user_configs=user_configs)

    def fit_round(self, server_round: int):
        """Perform a single round of federated training."""
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self.client_manager,
        )

        if not client_instructions:
            log(WARNING, "configure_fit: no clients selected, cancel")
            return None
        log(
            INFO,
            "configure_fit: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self.client_manager.num_available(),
        )

        # Start training of generator from current model parameters
        submitted_fs = {
            self.executor.submit(self.filter.server_tasks, self.parameters, server_round)
        }

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            executor=self.executor,
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            group_id=server_round,
        )
        log(
            INFO,
            "aggregate_fit: received %s results and %s failures",
            len(results),
            len(failures),
        )

        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

        # Perform the filteration on training results
        # self.filter.server_tasks(
        #     global_weights=self.parameters,
        #     server_round=server_round,
        # )
        (selected_indexes, client_stats) = self.filter.filter_updates(
            client_weights=[(fit_res.parameters, fit_res.num_examples) for _, fit_res in results],
            server_round=server_round,
        )

        # Aggregate training results
        aggregated_result: tuple[
            Optional[Parameters],
            dict[str, Scalar],
        ] = self.strategy.aggregate_fit(server_round=server_round, results=results, failures=failures, selected=selected_indexes)

        parameters_aggregated, metrics_aggregated = aggregated_result

        # Logging filtering stats of clients
        if self.filter.filter_type == "GAN" and client_stats is not None:
            metrics_aggregated["filter_loss"] = dict()
            metrics_aggregated["filter_accu_all"] = dict()
            metrics_aggregated["filter_accu_cls"] = dict()
        
            # Compile / collect results
            client_ids = [res.metrics["client_id"] for _, res in results]
            for index, cid in enumerate(client_ids):
                metrics_aggregated["filter_loss"][f"client_{cid}"] = client_stats["avg_loss"][index]
                metrics_aggregated["filter_accu_all"][f"client_{cid}"] = client_stats["accu_all"][index]
                metrics_aggregated["filter_accu_cls"][f"client_{cid}"] = client_stats["accu_cls"][index]

        if self.filter.filter_type == "KRUM":
            metrics_aggregated["distances"] = dict()

            # Compile / collect results
            client_ids = [res.metrics["client_id"] for _, res in results]
            for index, cid in enumerate(client_ids):
                metrics_aggregated["distances"][f"client_{cid}"] = client_stats["distances"][index, :]

        return parameters_aggregated, metrics_aggregated, (results, failures)
