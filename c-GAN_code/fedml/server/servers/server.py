"""Implementation of a Normal Server extending the built in FedML Server Class."""

import concurrent.futures
import copy
import timeit
from logging import DEBUG, INFO, WARNING, ERROR
from typing import Dict, Optional, List, Tuple, Union

from fedml.common.typing import Scalar, Parameters, Code, FitIns
from fedml.common.history import History
from fedml.common.logger import log

import os
import torch

from abc import ABC, abstractmethod

from fedml.defenses.create_filter import create_filter
from fedml.data_handler.data_split import CustomDataset
from fedml.data_handler.data_loader import load_data
from fedml.modules.strategy_functions import evaluate, get_evaluate_fn

class Server:
    def __init__(
        self, 
        *, 
        client_manager, 
        experiment_manager = None,
        strategy = None,
        user_configs: Dict ,
        initial_parameters = None,
        executor_type: str = "ThreadPool",
        num_gpus = None,
        
    ) -> None:
        
        self.experiment_manager = experiment_manager
        self.user_configs = user_configs
        self.client_manager = client_manager
        self.strategy = strategy
        
        self.set_initial_parameters(initial_parameters=initial_parameters)
        self._pre_attack_checkpoint_saved = False
        self._pre_attack_round = None
        self._first_round = 1
        try:
            attack_round = int(
                self.user_configs["EXPERIMENT_CONFIGS"]["MAL_HYPER_PARAM"]["ATTACK_ROUND"]
            )
            if attack_round and user_configs["MODEL_CONFIGS"]["WEIGHT_PATH"] is None:
                self._pre_attack_round = attack_round - 1
                self._first_round = 1
            elif attack_round:
                self._first_round = attack_round
            else:
                log(
                    WARNING,
                    "ATTACK_ROUND=%s has no positive pre-attack round; checkpointing disabled",
                    attack_round,
                )
        except Exception:
            log(
                WARNING,
                "ATTACK_ROUND not found in config; pre-attack checkpointing disabled",
            )
        # self.max_workers: Optional[int] = None
        self.max_workers: Optional[int] = self.strategy.min_fit_clients * 2 + 1 # +1 for the extra attack GAN just in case 

        self.executor = None
        if executor_type == "ThreadPool":
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        elif executor_type == "ProcessPool":
            self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers)
        
        # Distribute devices to attack and defense modules
        run_devices = [
            user_configs["SERVER_CONFIGS"]["FILTER_CONFIGS"]["HYPER_PARAM"]["TRAIN_GAN_PARAMS"]["DEVICE"],
            user_configs["EXPERIMENT_CONFIGS"]["MAL_HYPER_PARAM"]["GAN_ATTACK_CONFIG"]["HYPER_PARAM"]["TRAIN_GAN_PARAMS"]["DEVICE"],
        ]
        for i, device in enumerate(run_devices):
            if device in ["auto", "cuda"]:
                if num_gpus is not None:
                    run_devices[i] = f"cuda:{i % num_gpus}"
                else:
                    run_devices[i] = "cuda"
            else:
                run_devices[i] = "cpu"
        
        log(INFO, f"Filter creation, Run devices: {run_devices}, Num GPUs: {num_gpus}")


        #Create both filters
        self.filter = create_filter(user_configs=user_configs, mode="DEFENSE", device=run_devices[0]) 
        self.attack = create_filter(
            user_configs=user_configs,
            mode="ATTACK",
            device=run_devices[1],
        )
        self.attack.input_znoises = self.filter.input_znoises
        self.attack.input_classes = self.filter.input_classes
        

    

    def __del__(self):
        if self.executor:
            self.executor.shutdown()

    def set_initial_parameters(self, initial_parameters):
        if initial_parameters is not None:
            self.parameters = initial_parameters
        else:
            raise NotImplementedError("Parameter initialization not implemented")

    def _save_pre_attack_checkpoints(self, server_round: int) -> None:
        if self._pre_attack_checkpoint_saved:
            return
        if self._pre_attack_round is None or server_round != self._pre_attack_round:
            return

        output_path = self.user_configs["OUTPUT_CONFIGS"]["RESULT_LOG_PATH"]
        os.makedirs(output_path, exist_ok=True)
        model_name = self.user_configs["MODEL_CONFIGS"]["MODEL_NAME"]
        global_ckpt_path = os.path.join(
            output_path,
            f"weights-global-{model_name}-round-{server_round}_{self.filter.gen_model.gen_type}.pt",
        )
        global_weights = self.parameters
        if isinstance(global_weights, torch.Tensor):
            global_weights = global_weights.detach().clone().cpu()
        torch.save(obj=global_weights, f=global_ckpt_path)

        filter_ckpt_path = os.path.join(
            output_path,
            f"weights-gen-def_{self.filter.gen_model.gen_type}-round-{server_round}.pt",
        )
        filter_gen_model = getattr(self.filter, "gen_model", None)
        if filter_gen_model is not None:
            filter_gen_weights = filter_gen_model.get_weights()
            if isinstance(filter_gen_weights, torch.Tensor):
                filter_gen_weights = filter_gen_weights.detach().clone().cpu()
            torch.save(obj=filter_gen_weights, f=filter_ckpt_path)
            log(
                INFO,
                "Saved pre-attack checkpoints at round %s: global=%s, filter_generator=%s",
                server_round,
                global_ckpt_path,
                filter_ckpt_path,
            )
        else:
            log(
                WARNING,
                "Saved global pre-attack checkpoint at round %s to %s, but no filter generator was available",
                server_round,
                global_ckpt_path,
            )

        self._pre_attack_checkpoint_saved = True


    



    def fit_round(self, server_round: int):
        """Perform a single round of federated training."""
        # Get clients and their respective instructions from strategy
        # client_instructions = [(client, self.local_models[client.client_id], self.run_devices[client.client_id], fit_ins) for client in clients]
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self.client_manager,
        )

        mal_client_ins= [client_ins for client_ins in client_instructions if client_ins[0].client_type != "HONEST"]
        mal_cids = [client_ins[0].client_id for client_ins in mal_client_ins]

        if not client_instructions:
            log(WARNING, "configure_fit: no clients selected, cancel")
            return None

        mal_client_ins = [client_ins for client_ins in client_instructions if client_ins[0].client_type != "HONEST"]
        mal_cids = [client_ins[0].client_id for client_ins in mal_client_ins]

        log(
            INFO,
            "configure_fit: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self.client_manager.num_available(),
        )

        log(
            INFO,
            "SERVER ID: %s", 
            os.getpid()
        )

        #Start training filter gen from current model parameters (If empty not executed)
        
        filter_fs = self.filter.server_fit_round_before(
            global_parameters=self.parameters,
            server_round=server_round,
            executor=self.executor,
            client_instructions=client_instructions,
        )

        
        # Start training attack gen from current model parameters (If empty not executed)
        if self.attack is not None:
            attack_fs = self.attack.server_fit_round_before(
                global_parameters=self.parameters,
                server_round=server_round,
                executor=self.executor,
                client_instructions=client_instructions,
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

        finished_fs, _ = concurrent.futures.wait(  
            fs=filter_fs,
            timeout=None,  # Handled in the respective communication stack
        )


        ###############################################
        """
        This part is very relevant and fragile. Parent processes can't see child processes' memory, so we need to return the trained generator weights from the filter and attack modules and update them in the server after returning the future. 

        In the case of gan_attack, broadcast_gen_model already solves this by setting weights before finishing the task.

        It is essential to establish continuity between tasks (within the same task it's okay). All modifications of attributes   
        """
        # Collect trained generator weights from filter and update model---------------
        gen_weights = None
        for future in finished_fs:
            gen_weights = future.result()
            if gen_weights is not None:
                a=1
                # self.filter.gen_model.set_weights(gen_weights)

        mal_results = [res for res in results if res[0] in mal_cids] 

        gen_eval_fn=None

        if self.filter.filter_type == "GAN":
            filter_round, log_filter_stats, gen_def_dataset = self.filter.server_fit_round_after(gen_weights)
            selected_indexes, client_stats, gendef_dataset_2 = filter_round(server_round=server_round, results=results)
            metrics_filter = log_filter_stats(dict(), client_stats, results)

            try:
                filter_gen_weights = self.filter.gen_model.get_weights()
                filter_gen_device = next(self.filter.gen_model.parameters()).device
                same_weights = False
                same_device = False
                if gen_weights is not None:
                    same_weights = torch.equal(
                        filter_gen_weights.detach().cpu(),
                        gen_weights.detach().cpu() if isinstance(gen_weights, torch.Tensor) else gen_weights,
                    )
                    same_device = filter_gen_device == (
                        gen_weights.device if isinstance(gen_weights, torch.Tensor) else filter_gen_device
                    )
                log(
                    INFO,
                    "Filter generator weights/device match server_fit_round_after snapshot: same_weights=%s same_device=%s",
                    same_weights,
                    same_device,
                )
                if not same_weights:
                    log(INFO, "Filter generator weights differ from the server_fit_round_after snapshot")
                if not same_device:
                    log(INFO, "Filter generator device differs from the server_fit_round_after snapshot: filter=%s snapshot=%s", filter_gen_device, gen_weights.device if isinstance(gen_weights, torch.Tensor) else filter_gen_device)
            except Exception:
                log(INFO, "Filter generator weights/device match server_fit_round_after snapshot: %s %s", False, False)

            # Quick value equality check between the two generated datasets--------------
            try:
                ta = getattr(gen_def_dataset, "tensors", None)
                tb = getattr(gendef_dataset_2, "tensors", None)
                datasets_match = False
                if ta is not None and tb is not None and len(ta) == len(tb):
                    datasets_match = all(torch.equal(a, b) for a, b in zip(ta, tb))
                    any_diff = any((a-b).abs().max() > 1e-5 for a, b in zip(ta, tb))
                log(INFO, "Synthetic datasets identical: %s", datasets_match)
            except Exception:
                log(INFO, "Synthetic datasets identical: %s", False)
            #-------------------------------------------------
            gen_eval_fn = get_evaluate_fn(
            testset=gen_def_dataset,
            model_configs=self.filter.dis_configs,
            device=self.filter.device, # self.strategy.run_devices[0]
        )
        else: # Not using a filter
            metrics_filter = dict()
            selected_indexes = None 

        if self.attack is not None:
            eval_gen_attack_round, log_gen_attack_stats = self.attack.server_fit_round_after()
            _, attack_client_stats = eval_gen_attack_round(server_round=server_round, results=results)
            metrics_attack = log_gen_attack_stats(dict(), attack_client_stats, mal_results=mal_results)
        else: # Not using an attack
            metrics_attack = dict()

        # Evaluate mal_results across stages:
        log(
            INFO,
            "Evaluating MAL clients across stages"
            )    
        for client_id, res in mal_results:
            for stage, params in res.param_array.items():
                res_global = self.strategy.evaluate(server_round, parameters=params)
                loss_global = float("nan")
                metrics_global= dict()

                res_gen = (float("nan"), {})
                if res_global is not None:
                    loss_global, metrics_global = res_global
                    
                if gen_eval_fn is not None:
                        res_gen = gen_eval_fn(server_round, weights=params, config= {})

                loss_gen, metrics_gen = res_gen
                loss_combined= [loss_global, loss_gen]
                metrics_combined= [metrics_global, metrics_gen]
                log(
                    INFO,
                    "ID: %s, stage: %s , Global/GEN loss: %s, metrics: %s",
                    client_id,
                    stage,
                    loss_combined,
                    metrics_combined,
                )

            

        # self.eval_gen_attack_round, self.log_gen_attack_stats

        # Aggregate training results
        aggregated_result: tuple[
            Optional[Parameters],
            dict[str, Scalar],
        ] = self.strategy.aggregate_fit(server_round, results, failures, selected=selected_indexes)

        parameters_aggregated, metrics_aggregated = aggregated_result

        # Merge filter metrics with aggregated metrics
        metrics_aggregated.update(metrics_filter)
        metrics_aggregated.update(metrics_attack)

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
        for client_id, res in results:
            log(INFO, f"Eval Client (real): {client_id}:  metrics={res.metrics}")
            
            # client, model, device, ins in client_instructions

        # Aggregate the evaluation results
        aggregated_result: tuple[
            Optional[float],
            dict[str, Scalar],
        ] = self.strategy.aggregate_evaluate(server_round, results, failures)

        loss_aggregated, metrics_aggregated = aggregated_result
        train_asr = metrics_aggregated.get("train_asr", "NA") if isinstance(metrics_aggregated, dict) else "NA"
        test_asr = metrics_aggregated.get("test_asr", "NA") if isinstance(metrics_aggregated, dict) else "NA"
        log(INFO, "eval stats: (%s, %s)", train_asr, test_asr)

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

        log(INFO, f"Starting federated training for {num_rounds} rounds FROM ROUND {self._first_round} ...")
        for current_round in range(self._first_round, num_rounds+1):
            # Run a single fit round, collect results and update parameters
            # res_fit : parameters_aggregated, metrics_aggregated, (results, failures)
            # results : list of tuples (client_id, FitRes)
            res_fit = self.fit_round(current_round)

            if res_fit is not None:
                parameters_prime, fit_metrics, (results, failures) = res_fit
                if parameters_prime is not None:
                    self.parameters = parameters_prime
                if fit_metrics is None:
                    fit_metrics = dict()
                history.add_metrics_distributed_fit(server_round=current_round, metrics=fit_metrics)
                if self.experiment_manager is not None: self.experiment_manager.log(fit_metrics, nested=True)

            self._save_pre_attack_checkpoints(current_round)

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
    return client.client_id, fit_res

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
    return client.client_id, evaluate_res


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