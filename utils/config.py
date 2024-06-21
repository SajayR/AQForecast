import toml
from pathlib import Path

def load_config(config_path):
    with open(config_path, 'r') as f:
        return toml.load(f)

def load_data_config(identifier):
    config_path = Path(f"data_config/data_config_{identifier}.toml")
    return load_config(config_path)

def load_exp_config(identifier):
    config_path = Path(f"exp_config/exp_config_{identifier}.toml")
    return load_config(config_path)

def load_model_config(model_name, identifier):
    config_path = Path(f"models/{model_name}/config_{identifier}.toml")
    return load_config(config_path)