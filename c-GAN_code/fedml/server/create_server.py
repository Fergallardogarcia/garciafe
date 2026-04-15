"""A function to create desired type of FL server."""
from typing import Any

def create_server(
        server_type: str,
        client_manager,
    strategy: Any,
        user_configs: dict,
        executor_type: str,
        initial_parameters = None,
        experiment_manager = None,
        num_gpus = None,
    ):
    """Function to create the appropriat FL server instance."""
    
    assert server_type in ["NORMAL", "FILTER"], f"Invalid server {server_type} requested."

    if server_type in ["NORMAL", "FILTER"]:
        from .servers.server import Server
        return Server(
            client_manager=client_manager,
            strategy=strategy,
            experiment_manager=experiment_manager,
            initial_parameters=initial_parameters,
            user_configs=user_configs,
            executor_type=executor_type,
            num_gpus=num_gpus,
        )
        """
        elif server_type == "FILTER":
            from .servers.server_filter import FilterServer
            return FilterServer(
                client_manager=client_manager,
                strategy=strategy,
                experiment_manager=experiment_manager,
                initial_parameters=initial_parameters,
                user_configs=user_configs,
                executor_type=executor_type,
            )
        """
    else:
        raise ValueError(f"Invalid server {server_type} requested.")
