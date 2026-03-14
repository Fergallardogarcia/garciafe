"""A function to load the desired aggregation strategy."""

import math
import torch

from fedml.data_handler import load_data
from fedml.modules import (
    get_evaluate_fn,
    get_fit_config_fn,
    get_evaluate_config_fn,
    aggregate_fit_metrics,
    aggregate_evaluate_metrics,
)


def get_strategy(
        local_models, 
        run_devices,
        user_configs: dict,
    ):
    # Check what device to use on 
    # server side to run the computations
    run_device = run_devices[0]
    # run_device = ("cuda" if torch.cuda.is_available() else "cpu") \
    #     if user_configs["SERVER_CONFIGS"]["RUN_DEVICE"] == "auto" \
    #         else user_configs["SERVER_CONFIGS"]["RUN_DEVICE"]
    
    # Check wether to evaluate the global
    # model on the server side or not
    eval_fn = None
    if user_configs["SERVER_CONFIGS"]["EVALUATE_SERVER"]:
        # Load evaluation data
        _, testset = load_data(
            dataset_name=user_configs["DATASET_CONFIGS"]["DATASET_NAME"],
            dataset_path=user_configs["DATASET_CONFIGS"]["DATASET_PATH"],
            dataset_down=user_configs["DATASET_CONFIGS"]["DATASET_DOWN"]
        )

        eval_fn = get_evaluate_fn(
            testset=testset,
            model_configs=user_configs["MODEL_CONFIGS"],
            device=run_device
        )

    # Build the fit config function
    fit_config_fn = get_fit_config_fn(
        total_rounds=user_configs["SERVER_CONFIGS"]["NUM_TRAIN_ROUND"],
        local_epochs=user_configs["CLIENT_CONFIGS"]["LOCAL_EPCH"],
        lr_scheduler=user_configs["CLIENT_CONFIGS"]["LR_SCHEDULER"],
        scheduler_args=user_configs["CLIENT_CONFIGS"]["SCHEDULER_ARGS"],
        local_batchsize=user_configs["CLIENT_CONFIGS"]["BATCH_SIZE"],
        learning_rate=user_configs["CLIENT_CONFIGS"]["LEARN_RATE"],
        initial_lr=user_configs["CLIENT_CONFIGS"]["INITIAL_LR"],
        lr_warmup_steps=user_configs["CLIENT_CONFIGS"]["WARMUP_RDS"],
        optimizer_str=user_configs["CLIENT_CONFIGS"]["OPTIMIZER"],
        criterion_str=user_configs["CLIENT_CONFIGS"]["CRITERION"],
        perform_evals=user_configs["CLIENT_CONFIGS"]["EVALUATE"],
        optim_kwargs=user_configs["CLIENT_CONFIGS"]["OPTIM_ARG"],

    )
    
    evaluate_config_fn = get_evaluate_config_fn(
        total_rounds=user_configs["SERVER_CONFIGS"]["NUM_TRAIN_ROUND"],
        evaluate_bs=user_configs["CLIENT_CONFIGS"]["BATCH_SIZE"],
        criterion_str=user_configs["CLIENT_CONFIGS"]["CRITERION"],
    )

    # Get Strategy kwargs if any
    strategy_kwargs = user_configs["SERVER_CONFIGS"]["AGGR_STRAT_ARGS"] if user_configs["SERVER_CONFIGS"]["AGGR_STRAT_ARGS"] else dict()
    
    # Create an instance of the 
    # desired aggregation strategy
    if user_configs["SERVER_CONFIGS"]["AGGREGATE_STRAT"] == "FED-AVERAGE":
        from .strategies.federated_average import FederatedAverage
        stratgy = FederatedAverage(
            fraction_fit=user_configs["SERVER_CONFIGS"]["TRAINING_SAMPLE_FRACTION"],
            min_fit_clients=user_configs["SERVER_CONFIGS"]["MIN_TRAINING_SAMPLE_SIZE"],
            fraction_evaluate=user_configs["SERVER_CONFIGS"]["EVALUATE_SAMPLE_FRACTION"],
            min_evaluate_clients=user_configs["SERVER_CONFIGS"]["MIN_EVALUATE_SAMPLE_SIZE"],
            min_available_clients=user_configs["SERVER_CONFIGS"]["MIN_NUM_CLIENTS"],
            evaluate_fn=eval_fn,
            on_fit_config_fn=fit_config_fn,
            on_evaluate_config_fn=evaluate_config_fn,
            fit_metrics_aggregation_fn=aggregate_fit_metrics,
            evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
            local_models=local_models,
            run_devices=run_devices,
            **strategy_kwargs,
        )
        return stratgy
    elif user_configs["SERVER_CONFIGS"]["AGGREGATE_STRAT"] == "FED-MEDIAN":
        from .strategies.federated_median import FederatedMedian
        stratgy = FederatedMedian(
            fraction_fit=user_configs["SERVER_CONFIGS"]["TRAINING_SAMPLE_FRACTION"],
            min_fit_clients=user_configs["SERVER_CONFIGS"]["MIN_TRAINING_SAMPLE_SIZE"],
            fraction_evaluate=user_configs["SERVER_CONFIGS"]["EVALUATE_SAMPLE_FRACTION"],
            min_evaluate_clients=user_configs["SERVER_CONFIGS"]["MIN_EVALUATE_SAMPLE_SIZE"],
            min_available_clients=user_configs["SERVER_CONFIGS"]["MIN_NUM_CLIENTS"],
            evaluate_fn=eval_fn,
            on_fit_config_fn=fit_config_fn,
            on_evaluate_config_fn=evaluate_config_fn,
            fit_metrics_aggregation_fn=aggregate_fit_metrics,
            evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
            local_models=local_models,
            run_devices=run_devices,
            **strategy_kwargs,
        )
        return stratgy
    elif user_configs["SERVER_CONFIGS"]["AGGREGATE_STRAT"] == "FED-GEOMED":
        from .strategies.federated_geomed import FederatedGeometricMedian
        stratgy = FederatedGeometricMedian(
            fraction_fit=user_configs["SERVER_CONFIGS"]["TRAINING_SAMPLE_FRACTION"],
            min_fit_clients=user_configs["SERVER_CONFIGS"]["MIN_TRAINING_SAMPLE_SIZE"],
            fraction_evaluate=user_configs["SERVER_CONFIGS"]["EVALUATE_SAMPLE_FRACTION"],
            min_evaluate_clients=user_configs["SERVER_CONFIGS"]["MIN_EVALUATE_SAMPLE_SIZE"],
            min_available_clients=user_configs["SERVER_CONFIGS"]["MIN_NUM_CLIENTS"],
            evaluate_fn=eval_fn,
            on_fit_config_fn=fit_config_fn,
            on_evaluate_config_fn=evaluate_config_fn,
            fit_metrics_aggregation_fn=aggregate_fit_metrics,
            evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
            local_models=local_models,
            run_devices=run_devices,
            **strategy_kwargs,
        )
        return stratgy
    elif user_configs["SERVER_CONFIGS"]["AGGREGATE_STRAT"] == "FED-TRIMAVG":
        # Setup drop/cutout ratio if not already provided
        if "beta" not in strategy_kwargs.keys():
            strategy_kwargs["beta"] = user_configs["EXPERIMENT_CONFIGS"]["MAL_CLIENT_FRAC"]

        from .strategies.federated_trimmedavg import FederatedTrimmedAverage
        stratgy = FederatedTrimmedAverage(
            fraction_fit=user_configs["SERVER_CONFIGS"]["TRAINING_SAMPLE_FRACTION"],
            min_fit_clients=user_configs["SERVER_CONFIGS"]["MIN_TRAINING_SAMPLE_SIZE"],
            fraction_evaluate=user_configs["SERVER_CONFIGS"]["EVALUATE_SAMPLE_FRACTION"],
            min_evaluate_clients=user_configs["SERVER_CONFIGS"]["MIN_EVALUATE_SAMPLE_SIZE"],
            min_available_clients=user_configs["SERVER_CONFIGS"]["MIN_NUM_CLIENTS"],
            evaluate_fn=eval_fn,
            on_fit_config_fn=fit_config_fn,
            on_evaluate_config_fn=evaluate_config_fn,
            fit_metrics_aggregation_fn=aggregate_fit_metrics,
            evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
            local_models=local_models,
            run_devices=run_devices,
            **strategy_kwargs,
        )
        return stratgy
    else:
        raise ValueError(f"Invalid aggregation strategy {user_configs['SERVER_CONFIGS']['AGGREGATE_STRAT']} requested.")
