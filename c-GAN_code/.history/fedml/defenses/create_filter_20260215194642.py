"""A function to perform filtering of client updates with given modules."""

import math

from typing import Dict, TypedDict
from .filters.filter import Filter
from .filters.gan_filter import GenerativeFilter
from .filters.krum_filter import KrumFilter
from .filters.gan_attack import MaliciousGenerator

class FilterMap(TypedDict, total=False):
    GAN_FILTERING: GenerativeFilter
    KRUM_FILTERING: KrumFilter
    GAN_ATTACK: MaliciousGenerator


def create_filter(
        user_configs: Dict, mode: str = "d"
    ) -> FilterMap:
    """Create requested filter."""
    
    filters={}

    filter_type = user_configs["SERVER_CONFIGS"]["FILTER_CONFIGS"]["FILTER_TYPE"]
    filter_param = user_configs["SERVER_CONFIGS"]["FILTER_CONFIGS"]["HYPER_PARAM"]

    if filter_type == "GAN-FILTERING":
        from .filters.gan_filter import GenerativeFilter
        filters["GAN-FILTERING"] = GenerativeFilter(
            gen_configs=filter_param["GEN_ARGS"],
            dis_configs=user_configs["MODEL_CONFIGS"],
            train_configs=filter_param["TRAIN_GAN_PARAMS"],
            filter_configs=filter_param["FILTER_ARGS"],
            skip_rounds=filter_param["SKIP_ROUNDS"],
        )
    elif filter_type == "KRUM-FILTERING":
        from .filters.krum_filter import KrumFilter
        # Setup malicious client and to keep ratios if not already provided
        if "num_malicious_clients" not in filter_param.keys():
            filter_param["num_malicious_clients"] = math.ceil(user_configs["EXPERIMENT_CONFIGS"]["MAL_CLIENT_FRAC"] * user_configs["SERVER_CONFIGS"]["MIN_TRAINING_SAMPLE_SIZE"])
        if "num_clients_to_keep" not in filter_param.keys():
            filter_param["num_clients_to_keep"] = user_configs["SERVER_CONFIGS"]["MIN_TRAINING_SAMPLE_SIZE"] - filter_param["num_malicious_clients"]

        filters["KRUM-FILTERING"] = KrumFilter(
            **filter_param,
        )
    
    if user_configs["EXPERIMENT_CONFIGS"]["MAL_CLIENT_TYPE"] == "GAN_ATTACK":
        attack_type = user_configs["EXPERIMENT_CONFIGS"]["MAL_CLIENT_TYPE"]
        attack_param = user_configs["EXPERIMENT_CONFIGS"]["MAL_CLIENT_HYPER_PARAM"]["GAN_ATTACK_CONFIG"]

        if attack_type == "GAN_ATTACK":
            from .filters.gan_attack import MaliciousGenerator
            filters["GAN_ATTACK"] = MaliciousGenerator(
                gen_configs=attack_param["GEN_ARGS"],
                dis_configs=user_configs["MODEL_CONFIGS"],
                train_configs=attack_param["TRAIN_GAN_PARAMS"],
                filter_configs=attack_param["FILTER_ARGS"],
                skip_rounds=attack_param["SKIP_ROUNDS"],
            )
        return filters
    else:
        raise ValueError(f"Invalid filteration type {filter_type} requested.")
    
