"""Module to parse the configurations provided by user."""

import yaml
from yaml.constructor import SafeConstructor

def construct_yaml_tuple(self, node):
    seq = self.construct_sequence(node)
    # only make "leaf sequences" into tuples, you can add dict 
    # and other types as necessary
    if seq and isinstance(seq[0], (list, tuple)):
        return seq
    return tuple(seq)

SafeConstructor.add_constructor(
    u"tag:yaml.org,2002:python/tuple",
    construct_yaml_tuple
)

def parse_configs(config_path) -> dict:
    """Load and return user configurations."""
    with open(config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(exc)