"""A function to create desired type of FL server."""
from typing import Optional

# from flwr.client import Client

def create_client(
        client_type: Optional[str],
        client_id: int,
        trainset,
        testset,
        process: bool,
        configs: dict,
    ):
    """Function to create the appropriat FL server instance."""

    if (client_type == "HONEST") or (client_type is None):
        from .clients.honest_client import HonestClient
        return HonestClient(
            client_id=client_id,
            trainset=trainset,
            testset=testset,
            process=process,
        )
    elif client_type == "RANDOM":
        from .clients.malicious_random import RandomUpdateClient
        return RandomUpdateClient(
            client_id=client_id,
            trainset=trainset,
            testset=testset,
            process=process,
            attack_config = configs["MAL_HYPER_PARAM"],
        )
    elif client_type == "SIGNFLIP":
        from .clients.malicious_signflip import SignFlipClient
        return SignFlipClient(
            client_id=client_id,
            trainset=trainset,
            testset=testset,
            process=process,
            attack_config = configs["MAL_HYPER_PARAM"],
        )
    elif client_type == "LABELFLIP":
        from .clients.malicious_labelflip import LabelFlippingClient
        return LabelFlippingClient(
            client_id=client_id,
            trainset=trainset,
            testset=testset,
            process=process,
            attack_config = configs["MAL_HYPER_PARAM"],
        )
    elif client_type == "BACKDOOR":
        from .clients.malicious_backdoor import BackdoorClient
        return BackdoorClient(
            client_id=client_id,
            trainset=trainset,
            testset=testset,
            process=process,
            attack_config = configs["MAL_HYPER_PARAM"],
        )
    else:
        raise ValueError(f"Invalid server {client_type} requested.")
