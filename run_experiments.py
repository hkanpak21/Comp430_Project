#!/usr/bin/env python3
import os
import sys
import glob
import argparse
import subprocess
import json
import time
import datetime
from pathlib import Path
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='Run all SFL experiments')
    parser.add_argument('--configs_dir', type=str, default='configs', 
                        help='Directory containing configuration files')
    parser.add_argument('--output_dir', type=str, default='results', 
                        help='Directory to store results')
    parser.add_argument('--filter', type=str, default=None,
                        help='Only run configs matching this pattern (e.g., "mnist_clients5")')
    parser.add_argument('--max_parallel', type=int, default=1,
                        help='Maximum number of experiments to run in parallel')
    parser.add_argument('--timeout', type=int, default=7200,
                        help='Maximum time (in seconds) for each experiment')
    return parser.parse_args()

def run_experiment(config_path, output_dir):
    """Run a single experiment with the given config file"""
    config_name = os.path.basename(config_path).replace('.yaml', '')
    run_id = f"{config_name}_{int(time.time())}"
    
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting experiment: {config_name}")
    
    cmd = [
        "python", "experiments/train_secure_sfl.py",
        "--config", config_path,
        "--run_id", run_id
    ]
    
    try:
        # Run the experiment with timeout
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        stdout, stderr = process.communicate(timeout=args.timeout)
        
        # Create experiment output directory
        exp_dir = os.path.join(output_dir, config_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Save stdout and stderr
        with open(os.path.join(exp_dir, "stdout.log"), "w") as f:
            f.write(stdout)
        
        with open(os.path.join(exp_dir, "stderr.log"), "w") as f:
            f.write(stderr)
        
        # Copy metrics file to results directory
        metrics_path = os.path.join("experiments", "out", run_id, "metrics.json")
        if os.path.exists(metrics_path):
            shutil.copy(metrics_path, os.path.join(exp_dir, "metrics.json"))
            
            # Extract key metrics for summary
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                
            return {
                'config': config_name,
                'success': True,
                'test_accuracy': metrics.get('final_test_acc', None),
                'privacy_epsilon': metrics.get('privacy', {}).get('final_epsilon', None),
                'training_time': metrics.get('total_elapsed_time', None)
            }
        else:
            print(f"[ERROR] No metrics file found for {config_name}")
            return {
                'config': config_name,
                'success': False,
                'error': 'No metrics file generated'
            }
            
    except subprocess.TimeoutExpired:
        process.kill()
        print(f"[ERROR] Experiment {config_name} timed out after {args.timeout} seconds")
        return {
            'config': config_name,
            'success': False,
            'error': f'Timed out after {args.timeout} seconds'
        }
    except Exception as e:
        print(f"[ERROR] Failed to run experiment {config_name}: {str(e)}")
        return {
            'config': config_name,
            'success': False,
            'error': str(e)
        }

def main(args):
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all config files
    config_files = glob.glob(os.path.join(args.configs_dir, "*.yaml"))
    
    # Apply filter if provided
    if args.filter:
        config_files = [f for f in config_files if args.filter in os.path.basename(f)]
    
    # Sort config files for deterministic execution
    config_files.sort()
    
    print(f"Found {len(config_files)} configuration files to process")
    
    # Run experiments
    results = []
    for config_file in config_files:
        result = run_experiment(config_file, args.output_dir)
        results.append(result)
    
    # Generate summary report
    summary = {
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_experiments': len(config_files),
        'successful_experiments': sum(1 for r in results if r.get('success', False)),
        'failed_experiments': sum(1 for r in results if not r.get('success', False)),
        'results': results
    }
    
    # Save summary
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nExperiment Summary:")
    print(f"Total experiments: {summary['total_experiments']}")
    print(f"Successful: {summary['successful_experiments']}")
    print(f"Failed: {summary['failed_experiments']}")
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    args = parse_args()
    main(args) 