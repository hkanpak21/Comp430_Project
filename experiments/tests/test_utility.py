import yaml
from .utils import run_experiment

def test_minimum_accuracy():
    """Tests that the model achieves the minimum accuracy specified in the config file."""
    # First load the config to get the minimum accuracy threshold
    config_path = "configs/dnn_default.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set a default min_acc if not specified in the config
    min_acc = config.get('min_acc', 0.7)  # Default to 70% if not specified
    
    # Run the experiment
    m = run_experiment(config_path)
    
    # Check if accuracy meets or exceeds the minimum threshold
    assert m["final_test_acc"] >= min_acc, f"Model accuracy {m['final_test_acc']} below minimum threshold {min_acc}"
