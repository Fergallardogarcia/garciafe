"""A function to perform filtering of client updates with given modules."""

import math

from typing import Dict, Optional, TypedDict
from .filters.filter import Filter
from .filters.gan_filter import GenerativeFilter
from .filters.krum_filter import KrumFilter
from fedml.attacks import GAN_attack

class FilterMap(TypedDict, total=False):
    GAN_FILTERING: GenerativeFilter
    KRUM_FILTERING: KrumFilter
    GAN_ATTACK: GAN_attack


def create_filter(
        user_configs: Dict, 
        device: Optional[str] = "cuda",
        mode: str = "DEFENSE"
    ) -> Filter:
    """Create requested filter."""
    
    filter_type = user_configs["SERVER_CONFIGS"]["FILTER_CONFIGS"]["FILTER_TYPE"]
    filter_param = user_configs["SERVER_CONFIGS"]["FILTER_CONFIGS"]["HYPER_PARAM"]
    if user_configs["SERVER_CONFIGS"]["SERVER_TYPE"] == "FILTER":
        
        if mode == "DEFENSE":
            if filter_type == "GAN-FILTERING":
                from .filters.gan_filter import GenerativeFilter
                return GenerativeFilter(
                    gen_configs=filter_param["GEN_ARGS"],
                    dis_configs=user_configs["MODEL_CONFIGS"],
                    train_configs=filter_param["TRAIN_GAN_PARAMS"],
                    filter_configs=filter_param["FILTER_ARGS"],
                    skip_rounds=filter_param["SKIP_ROUNDS"],
                    cuda_device=device,
                )
            elif filter_type == "KRUM-FILTERING":
                from .filters.krum_filter import KrumFilter
                # Setup malicious client and to keep ratios if not already provided
                if "num_malicious_clients" not in filter_param.keys():
                    filter_param["num_malicious_clients"] = math.ceil(user_configs["EXPERIMENT_CONFIGS"]["MAL_CLIENT_FRAC"] * user_configs["SERVER_CONFIGS"]["MIN_TRAINING_SAMPLE_SIZE"])
                if "num_clients_to_keep" not in filter_param.keys():
                    filter_param["num_clients_to_keep"] = user_configs["SERVER_CONFIGS"]["MIN_TRAINING_SAMPLE_SIZE"] - filter_param["num_malicious_clients"]

                return KrumFilter(
                    **filter_param,
                )
        
    if mode == "ATTACK":
        attack_type = user_configs["EXPERIMENT_CONFIGS"]["MAL_CLIENT_TYPE"]
        attack_param = user_configs["EXPERIMENT_CONFIGS"]["MAL_HYPER_PARAM"]["GAN_ATTACK_CONFIG"]

        if attack_type == "GAN_ATTACK":
            from fedml.attacks import GAN_attack
            return GAN_attack(
                gen_configs=attack_param["HYPER_PARAM"]["GEN_ARGS"],
                dis_configs=user_configs["MODEL_CONFIGS"],
                train_configs=attack_param["HYPER_PARAM"]["TRAIN_GAN_PARAMS"],
                filter_configs=filter_param["FILTER_ARGS"], #Sloppy because we are using unnecesary info to initialize. Fix later
                skip_rounds=user_configs["EXPERIMENT_CONFIGS"]["MAL_HYPER_PARAM"]["ATTACK_ROUND"],
                cuda_device=device,
            )
        
    else:
        return Filter()
        #Value errors should be adapted for wrong mode ("ATTACK" or "DEFENSE") as well as wrong filter ("GAN-FILTERING", "KRUM-FILTERING") or attack type
        # raise ValueError(f"Invalid filteration type {filter_type} requested.")  
    
