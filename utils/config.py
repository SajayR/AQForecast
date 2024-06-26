import toml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return toml.load(f)
