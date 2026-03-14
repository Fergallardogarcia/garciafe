"""Module to parse the configurations provided by user."""

import yaml

def store_configs(data_dict, config_path) -> None:
    """Save user configurations to file."""
    with open(config_path, "w") as outfile:
        try:
            yaml.dump(data_dict, outfile, default_flow_style=False)
        except yaml.YAMLError as exc:
            print(exc)
