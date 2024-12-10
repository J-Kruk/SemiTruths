import os
import yaml
import pandas as pd


def load_config_file(filepath):
    with open(filepath, "r") as file:
        return yaml.safe_load(file)


def merge_args_with_config(args, config):
    for key, value in config.items():
        # if not getattr(args, key, None):
        setattr(args, key, value)
    return args
