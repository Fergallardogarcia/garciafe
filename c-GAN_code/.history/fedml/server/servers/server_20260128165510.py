"""Implementation of a Normal Server extending the built in FedML Server Class."""

import concurrent.futures
import timeit
from logging import DEBUG, INFO, WARNING, ERROR
from typing import Dict, Optional, List, Tuple, Union

from fedml.common.typing import Scalar, Parameters, Code
from fedml.common.history import History
from fedml.common.logger import log


class Server:
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
        
        self.experiment_manager = experiment_manager
        self.user_configs = user_configs
        self.client_manager = client_manager
        self.strategy = strategy
        self.set_initial_parameters(initial_parameters=initial_parameters)
        # self.max_workers: Optional[int] = None
        self.max_workers: Optional[int] = self.strategy.min_fit_clients * 2

        self.executor = None
        if executor_type == "ThreadPool":
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        elif executor_type == "ProcessPool":
            self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers)

    def __del__(self):
        if self.executor:
            self.executor.shutdown()

    def set_initial_parameters(self, initial_parameters):
        if initial_parameters is not None:
            self.parameters = initial_parameters
        else:
            raise NotImplementedError("Parameter initialization not implemented")

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

        # Aggregate training results
        aggregated_result: tuple[
            Optional[Parameters],
            dict[str, Scalar],
        ] = self.strategy.aggregate_fit(server_round, results, failures)

        parameters_aggregated, metrics_aggregated = aggregated_result
        return parameters_aggregated, metrics_aggregated, (results, failures)


    def evaluate_round(self, server_round: int):
        """Validate current global model on a number of clients."""
        client_instructions = self.strategy.configure_evaluate(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self.client_manager,
        )
        if not client_instructions:
            log(WARNING, "configure_evaluate: no clients selected, skipping evaluation")
            return None
        log(
            INFO,
            "configure_evaluate: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self.client_manager.num_available(),
        )

        # Collect `evaluate` results from all clients participating in this round
        results, failures = evaluate_clients(
            executor=self.executor,
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            group_id=server_round,
        )
        log(
            INFO,
            "aggregate_evaluate: received %s results and %s failures",
            len(results),
            len(failures),
        )

        # Aggregate the evaluation results
        aggregated_result: tuple[
            Optional[float],
            dict[str, Scalar],
        ] = self.strategy.aggregate_evaluate(server_round, results, failures)

        loss_aggregated, metrics_aggregated = aggregated_result
        log(
            INFO,
            "eval stats: (%s, %s)",
            metrics_aggregated["train_asr"],
            metrics_aggregated["test_asr"],
        )

        return loss_aggregated, metrics_aggregated, (results, failures)


    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int):
        """Run federated training for a number of rounds."""
        history = History()

        # Initial Evaluation
        log(INFO, "[INIT]")
        log(INFO, "Starting evaluation of initial global parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])

        # Run federated training for num_rounds
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds+1):
            # Run a single fit round, collect results and update parameters
            res_fit = self.fit_round(current_round)

            if res_fit is not None:
                parameters_prime, fit_metrics, _ = res_fit
                if parameters_prime is not None:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(server_round=current_round, metrics=fit_metrics)
                if self.experiment_manager is not None: self.experiment_manager.log(fit_metrics, nested=True)

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

                if self.experiment_manager is not None:                     
                    logging_metrics = {
                        "centralized_loss": loss_cen,
                        "centralized_accu": metrics_cen["accuracy"],
                    }
                    self.experiment_manager.log(logging_metrics, nested=True)

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=current_round)
            if res_fed is not None:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed is not None:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )
                if self.experiment_manager is not None: self.experiment_manager.log(evaluate_metrics_fed, nested=False)

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        return history, elapsed

def fit_clients(executor, client_instructions, max_workers: Optional[int], group_id: int):
    """Refine parameters concurrently on all selected clients."""
    
    # with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    # with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    submitted_fs = {
        executor.submit(fit_client, client, model, device, ins, group_id) for client, model, device, ins in client_instructions
    }
    finished_fs, _ = concurrent.futures.wait(
        fs=submitted_fs,
        timeout=None,  # Handled in the respective communication stack
    )

    # Gather results
    results = []
    failures = []
    for future in finished_fs:
        _handle_finished_future_after_fit(future=future, results=results, failures=failures)
    return results, failures

def fit_client(client, model, device, ins, group_id: int):
    """Refine parameters on a single client."""
    fit_res = client.fit(model, device, ins) #, group_id=group_id)
    return client, fit_res

def _handle_finished_future_after_fit(
        future: concurrent.futures.Future,  # type: ignore
        results,
        failures,
    ) -> None:
    """Convert finished future into either a result or a failure."""
    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        import traceback
        log(ERROR, str(failure) + "\n" + "".join(traceback.format_exception(failure)))
        failures.append(failure)
        return

    # Successfully received a result from a client
    result = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)

def evaluate_clients(executor, client_instructions, max_workers: Optional[int], group_id: int,):
    """Evaluate parameters concurrently on all selected clients."""

    # with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    # with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    submitted_fs = {
        executor.submit(evaluate_client, client, model, device, ins, group_id)
        for client, model, device, ins in client_instructions
    }
    finished_fs, _ = concurrent.futures.wait(
        fs=submitted_fs,
        timeout=None,  # Handled in the respective communication stack
    )

    # Gather results
    results = []
    failures = []
    for future in finished_fs:
        _handle_finished_future_after_evaluate(
            future=future, results=results, failures=failures
        )
    return results, failures


def evaluate_client(client, model, device, ins, group_id: int):
    """Evaluate parameters on a single client."""
    evaluate_res = client.evaluate(model, device, ins)
    return client, evaluate_res


def _handle_finished_future_after_evaluate(
    future: concurrent.futures.Future,  # type: ignore
    results,
    failures,
) -> None:
    """Convert finished future into either a result or a failure."""
    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        import traceback
        log(ERROR, str(failure) + "\n" + "".join(traceback.format_exception(failure)))
        failures.append(failure)
        return

    # Successfully received a result from a client
    result = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)
