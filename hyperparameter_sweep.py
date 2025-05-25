#!/usr/bin/env python3
"""
Hyperparameter sweep for Secure Split-FL to achieve higher accuracy.
This script runs multiple training configurations and logs results to a file.
"""

import os
import sys
import yaml
import json
import time
import pathlib
import subprocess
import itertools
from datetime import datetime

# Configuration parameters to sweep
SWEEP_PARAMS = {
    # Model architecture options
    "model": ["SimpleDNN", "SimpleCNN"],
    
    # Training parameters
    "num_clients": [1, 5, 10, 20],
    "batch_size": [32, 64, 128],
    "num_rounds": [20, 50, 100],
    "lr": [0.001, 0.01, 0.05, 0.1],
    
    # Split layer options (depends on model)
    # For SimpleDNN: 0=Flatten, 1=fc1, 2=relu1, 3=fc2, 4=relu2
    # For SimpleCNN: 0=conv1, 1=relu1, 2=pool1, 3=conv2, 4=relu2, 5=pool2, 6=flatten
    "cut_layer": [2, 4, 6],  # Will filter incompatible options later
    
    # DP noise parameters
    "initial_sigma": [0.0, 0.5, 1.0],
    "adaptive_noise_decay_factor": [0.99, 0.995, 0.999]
}

# Fixed parameters (not swept)
FIXED_PARAMS = {
    "seed": 42,
    "dataset": "MNIST",
    "data_dir": "./data",
    "optimizer": "SGD",  # Only SGD allowed
    "dp_noise": {
        "laplacian_sensitivity": 0.0,
        "epsilon_prime": 1.0,
        "clip_norm": 1.0,
        "noise_multiplier": 0.0,
        "delta": 1e-5,
        "adaptive_clipping_factor": 1.0,
        "validation_set_ratio": 0.1,
        "noise_decay_patience": 3
    }
}

def create_config(params):
    """Create a configuration dictionary from parameters."""
    config = FIXED_PARAMS.copy()
    
    # Update with sweep parameters
    for key, value in params.items():
        if key == "initial_sigma" or key == "adaptive_noise_decay_factor":
            config["dp_noise"][key] = value
        else:
            config[key] = value
    
    return config

def is_valid_config(params):
    """Check if the configuration is valid."""
    # Check model and cut_layer compatibility
    if params["model"] == "SimpleDNN" and params["cut_layer"] > 4:
        return False
    if params["model"] == "SimpleCNN" and params["cut_layer"] < 2:
        return False
    
    return True

def run_experiment(config, output_dir, run_id):
    """Run a single experiment with the given configuration."""
    # Create config file
    config_path = os.path.join(output_dir, f"config_{run_id}.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Run the experiment
    cmd = [
        "python", "experiments/train_secure_sfl.py",
        "--config", config_path,
        "--run_id", run_id
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed_time = time.time() - start_time
    
    # Extract final accuracy from stdout
    accuracy = None
    for line in result.stdout.split('\n'):
        if "Final Test Accuracy:" in line:
            try:
                accuracy = float(line.split(':')[1].strip().replace('%', '')) / 100.0
            except:
                pass
    
    # Check if metrics.json exists
    metrics_path = os.path.join("experiments", "out", run_id, "metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    
    return {
        "run_id": run_id,
        "success": result.returncode == 0,
        "accuracy": accuracy,
        "elapsed_time": elapsed_time,
        "metrics": metrics
    }

def main():
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("sweep_results", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create results file
    results_file = os.path.join(output_dir, "results.txt")
    with open(results_file, 'w') as f:
        f.write("Secure Split-FL Hyperparameter Sweep Results\n")
        f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("=" * 80 + "\n\n")
    
    # Generate parameter combinations
    param_names = list(SWEEP_PARAMS.keys())
    param_values = list(SWEEP_PARAMS.values())
    
    # Filter to a reasonable number of combinations (max 50)
    all_combinations = list(itertools.product(*param_values))
    valid_combinations = [dict(zip(param_names, combo)) for combo in all_combinations if is_valid_config(dict(zip(param_names, combo)))]
    
    # Prioritize combinations that are more likely to succeed
    # Sort by: model (CNN first), higher rounds, lower initial_sigma
    valid_combinations.sort(key=lambda x: (
        0 if x["model"] == "SimpleCNN" else 1,
        -x["num_rounds"],
        x["initial_sigma"]
    ))
    
    # Limit to 50 combinations
    selected_combinations = valid_combinations[:50]
    
    print(f"Running {len(selected_combinations)} configurations")
    
    # Run experiments
    for i, params in enumerate(selected_combinations):
        print(f"\nRunning configuration {i+1}/{len(selected_combinations)}")
        print(f"Parameters: {params}")
        
        config = create_config(params)
        run_id = f"sweep_{timestamp}_{i:03d}"
        
        result = run_experiment(config, output_dir, run_id)
        
        # Log result
        with open(results_file, 'a') as f:
            f.write(f"Configuration {i+1}/{len(selected_combinations)}\n")
            f.write(f"Parameters: {params}\n")
            f.write(f"Success: {result['success']}\n")
            f.write(f"Accuracy: {result['accuracy']}\n")
            f.write(f"Time: {result['elapsed_time']:.2f} seconds\n")
            if result['metrics']:
                f.write(f"Epsilon: {result['metrics'].get('epsilon')}\n")
                f.write(f"Final Sigma: {result['metrics'].get('sigma')}\n")
            f.write("\n" + "-" * 50 + "\n\n")
        
        # Also print to console
        print(f"Success: {result['success']}")
        print(f"Accuracy: {result['accuracy']}")
        print(f"Time: {result['elapsed_time']:.2f} seconds")
    
    # Write summary
    with open(results_file, 'a') as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Sweep completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total configurations run: {len(selected_combinations)}\n")
    
    print(f"\nSweep completed. Results saved to {results_file}")
    
    # Find best configuration
    best_accuracy = 0
    best_config = None
    
    for i, params in enumerate(selected_combinations):
        run_id = f"sweep_{timestamp}_{i:03d}"
        metrics_path = os.path.join("experiments", "out", run_id, "metrics.json")
        
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                accuracy = metrics.get("final_test_acc", 0)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_config = params
    
    if best_config:
        print(f"\nBest configuration:")
        print(f"Accuracy: {best_accuracy:.4f}")
        print(f"Parameters: {best_config}")
        
        with open(results_file, 'a') as f:
            f.write("\nBest configuration:\n")
            f.write(f"Accuracy: {best_accuracy:.4f}\n")
            f.write(f"Parameters: {best_config}\n")

if __name__ == "__main__":
    main()
