#!/usr/bin/env python3
import os
import sys
import glob
import time
import subprocess
import csv
import json
import yaml
from datetime import datetime
from tqdm import tqdm

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def run_experiment(config_path, output_file):
    """Run an experiment with the given configuration file"""
    start_time = time.time()
    print(f"\nRunning experiment with config: {config_path}")
    
    # Command to run the training script
    cmd = f"python experiments/train_secure_sfl.py --config {config_path}"
    
    # Capture output
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    output = result.stdout
    error = result.stderr
    
    elapsed_time = time.time() - start_time
    
    # Extract final accuracy from output - try multiple patterns
    final_acc = None
    
    # Pattern 1: "Final test accuracy: XX.XX%"
    for line in output.split('\n'):
        if 'Final test accuracy' in line:
            try:
                final_acc = float(line.split(':')[-1].strip().rstrip('%'))
                break
            except:
                pass
    
    # Pattern 2: "Test accuracy: XX.XX%" (last occurrence)
    if final_acc is None:
        acc_lines = [line for line in output.split('\n') if 'Test accuracy' in line]
        if acc_lines:
            try:
                final_acc = float(acc_lines[-1].split(':')[-1].strip().rstrip('%'))
            except:
                pass
    
    # Pattern 3: "Accuracy: XX.XX%" (last occurrence)
    if final_acc is None:
        acc_lines = [line for line in output.split('\n') if 'Accuracy:' in line]
        if acc_lines:
            try:
                final_acc = float(acc_lines[-1].split(':')[-1].strip().rstrip('%'))
            except:
                pass
    
    # Prepare result summary
    config_name = os.path.basename(config_path)
    result_summary = f"\n{'='*50}\n"
    result_summary += f"Config: {config_name}\n"
    result_summary += f"Time: {elapsed_time:.2f} seconds\n"
    
    if final_acc is not None:
        result_summary += f"Final Accuracy: {final_acc:.2f}%\n"
    else:
        result_summary += "Final Accuracy: Not found in output\n"
        
        # Save debug output
        debug_file = f"debug_{config_name}.txt"
        with open(debug_file, 'w') as f:
            f.write(f"STDOUT:\n{output}\n\nSTDERR:\n{error}")
        result_summary += f"Debug output saved to: {debug_file}\n"
    
    # Check for errors
    if error:
        result_summary += f"\nErrors:\n{error}\n"
    
    # Save to output file
    with open(output_file, 'a') as f:
        f.write(result_summary)
    
    print(result_summary)
    
    # Load config details
    try:
        config = load_config(config_path)
    except:
        config = {}
    
    return {
        'config_path': config_path,
        'config_name': config_name,
        'final_acc': final_acc,
        'elapsed_time': elapsed_time,
        'success': final_acc is not None and not error,
        'error': error if error else None,
        'model': config.get('model', 'Unknown'),
        'optimizer': config.get('optimizer', 'Unknown'),
        'cut_layer': config.get('cut_layer', 'Unknown'),
        'lr': config.get('lr', 'Unknown'),
        'batch_size': config.get('batch_size', 'Unknown'),
        'num_clients': config.get('num_clients', 'Unknown'),
        'num_rounds': config.get('num_rounds', 'Unknown'),
        'initial_sigma': config.get('dp_noise', {}).get('initial_sigma', 'Unknown'),
        'noise_multiplier': config.get('dp_noise', {}).get('noise_multiplier', 'Unknown')
    }

def export_to_csv(results, csv_file):
    """Export results to CSV file"""
    fieldnames = [
        'config_name', 'model', 'optimizer', 'cut_layer', 'lr', 'batch_size', 
        'num_clients', 'num_rounds', 'initial_sigma', 'noise_multiplier',
        'final_acc', 'elapsed_time', 'success'
    ]
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            # Filter to only include the fields we want
            filtered_result = {k: v for k, v in result.items() if k in fieldnames}
            writer.writerow(filtered_result)
    
    print(f"Results exported to {csv_file}")

def main():
    # Set up experiment parameters
    config_filter = "exp_"  # Filter to only run our experimental configs
    report_file = "experiment_results.txt"
    csv_file = "experiment_results.csv"
    
    # Get all config files
    config_files = glob.glob("configs/*.yaml")
    
    # Apply filter if provided
    if config_filter:
        config_files = [f for f in config_files if config_filter in f]
    
    print(f"Found {len(config_files)} configuration files to test")
    
    # Create report file
    with open(report_file, 'w') as f:
        f.write(f"# Secure SFL Experiment Report\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Run experiments
    results = []
    for i, config_path in enumerate(sorted(config_files), 1):
        print(f"\nExperiment {i}/{len(config_files)}")
        result = run_experiment(config_path, report_file)
        results.append(result)
    
    # Export results to CSV
    export_to_csv(results, csv_file)
    
    # Generate summary
    successful_results = [r for r in results if r['success']]
    
    with open(report_file, 'a') as f:
        f.write(f"\n\n## Summary of Results\n\n")
        
        # Overall statistics
        f.write(f"### Overall Statistics\n\n")
        f.write(f"Total experiments run: {len(results)}\n")
        f.write(f"Successful experiments: {len(successful_results)}\n")
        f.write(f"Success rate: {len(successful_results)/len(results)*100:.2f}%\n\n")
        
        # Best configurations
        f.write(f"### Top Configurations by Accuracy\n\n")
        best_configs = sorted(successful_results, key=lambda x: x['final_acc'] if x['final_acc'] else 0, reverse=True)
        
        f.write(f"| Rank | Configuration | Model | Cut Layer | Optimizer | LR | Batch Size | Clients | Accuracy (%) | Runtime (s) |\n")
        f.write(f"|------|--------------|-------|-----------|-----------|----|-----------|---------|--------------|--------------|\n")
        
        for i, result in enumerate(best_configs, 1):
            f.write(f"| {i} | {result['config_name']} | {result['model']} | {result['cut_layer']} | {result['optimizer']} | {result['lr']} | {result['batch_size']} | {result['num_clients']} | {result['final_acc']:.2f} | {result['elapsed_time']:.2f} |\n")
    
    print(f"\nExperiment report generated: {report_file}")
    print(f"Results exported to CSV: {csv_file}")
    print(f"Total experiments: {len(results)}")
    print(f"Successful experiments: {len(successful_results)}")
    
    if successful_results:
        best_result = max(successful_results, key=lambda x: x['final_acc'] if x['final_acc'] else 0)
        print(f"\nBest configuration: {best_result['config_name']}")
        print(f"Best accuracy: {best_result['final_acc']:.2f}%")
        print(f"Model: {best_result['model']}, Cut Layer: {best_result['cut_layer']}")
        print(f"Optimizer: {best_result['optimizer']}, Learning Rate: {best_result['lr']}")
        print(f"Batch Size: {best_result['batch_size']}, Clients: {best_result['num_clients']}")

if __name__ == "__main__":
    main()
