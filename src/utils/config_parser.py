import yaml
import argparse
import os

def load_config(config_path):
    """Loads configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='Secure Split Federated Learning Simulation')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file.')
    args = parser.parse_args()
    return args

def get_config():
    """Parses args and loads the specified config file."""
    args = parse_args()
    config = load_config(args.config)
    # You might want to add schema validation or default value handling here
    return config 

def get_config_from_file(config_path):
    """Loads configuration directly from a specified file path."""
    config = load_config(config_path)
    return config