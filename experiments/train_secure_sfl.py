#!/usr/bin/env python3
import sys
import os
import argparse
import json
import pathlib
# Add the project root directory (Comp430_Project) to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import numpy as np
import random
import time
import copy
from collections import OrderedDict

# Imports for flattened structure
from src.utils.config_parser import get_config_from_file as get_config
from src.datasets.data_loader import get_mnist_dataloaders, get_client_data_loaders
from src.models import get_model # Import from models package
from src.models.split_utils import split_model, get_combined_model
from src.dp.privacy_accountant import ManualPrivacyAccountant
from src.dp.registry import get_accountant, HybridAccountant
from src.sfl.client import SFLClient
from src.sfl.main_server import MainServer
from src.sfl.fed_server import FedServer

def set_seed(seed):
    """Sets random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate_model(model, test_loader, device):
    """Evaluates the model on the test dataset."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def parse_args():
    parser = argparse.ArgumentParser(description='Secure Split Federated Learning')
    parser.add_argument('--config', type=str, default='configs/dnn_default.yaml', help='Path to config file')
    parser.add_argument('--run_id', type=str, default=None, help='Run ID for storing results')
    return parser.parse_args()

def main():
    args = parse_args()
    config = get_config(config_path=args.config)
    
    # Setup output directory for metrics
    out_dir = None
    if args.run_id:
        out_dir = pathlib.Path(__file__).parent / "out" / args.run_id
        out_dir.mkdir(parents=True, exist_ok=True)
    print("Configuration loaded:", config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(config['seed'])

    # Get train, validation, and test loaders
    train_loader, test_loader, train_dataset, _ = get_mnist_dataloaders(config)
    
    # Split training data into client data and validation set
    validation_ratio = config['dp_noise']['validation_set_ratio']
    validation_size = int(len(train_dataset) * validation_ratio)
    train_size = len(train_dataset) - validation_size
    
    train_dataset, validation_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, validation_size]
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )
    
    client_loaders = get_client_data_loaders(config, train_dataset)
    num_clients = config['num_clients']
    print(f"Loaded {config['dataset']} dataset and created {num_clients} client data loaders.")
    print(f"Created validation set with {len(validation_dataset)} samples.")

    full_model = get_model(config['model']).to(device) # Use the get_model function
    client_model_template, server_model_template = split_model(full_model, config['cut_layer'])
    print(f"Split model '{config['model']}' at layer index {config['cut_layer']}.")
    print("Client model part:", client_model_template)
    print("Server model part:", server_model_template)

    fed_server = FedServer(copy.deepcopy(client_model_template), config, device)
    main_server = MainServer(copy.deepcopy(server_model_template), config, device)

    clients = []
    for i in range(num_clients):
        client_model_copy = copy.deepcopy(client_model_template)
        client = SFLClient(client_id=i, client_model=client_model_copy, dataloader=client_loaders[i], config=config, device=device)
        clients.append(client)
    print(f"Initialized {len(clients)} SFL clients.")

    # Calculate sampling rate
    dataset_size = len(train_dataset)
    batch_size = config['batch_size']
    sampling_rate = batch_size / dataset_size
    
    # Use the accountant factory instead of direct initialization
    acc_cfg = config['dp_noise']
    privacy_accountant = get_accountant(
        acc_cfg.get('mode', 'hybrid'),  # Default to hybrid accountant
        noise_multiplier=acc_cfg['initial_sigma'],
        sampling_rate=sampling_rate,
        moment_orders=None
    )
    target_delta = float(config['dp_noise']['delta'])
    print(f"Initialized Privacy Accountant in {acc_cfg.get('mode', 'hybrid')} mode. Sampling rate (q): {sampling_rate:.4f}, Target Delta: {target_delta}")

    start_time = time.time()
    num_rounds = config['num_rounds']
    print(f"\nStarting SFLV1 training for {num_rounds} rounds...")

    for round_num in range(num_rounds):
        round_start_time = time.time()
        print(f"\n--- Round {round_num + 1}/{num_rounds} ---")
        
        # Broadcast current noise scale to clients
        current_sigma = fed_server.get_current_sigma()
        for client in clients:
            client.update_noise_scale(current_sigma)
        
        global_client_params = fed_server.get_client_model_params()
        for client in clients:
            client.set_model_params(global_client_params)

        client_activation_grads = {}
        client_noisy_wc_grads = []
        main_server.clear_round_data()

        client_data_for_main_server = {}
        for client in clients:
            noisy_activations, labels = client.local_forward_pass()
            client_data_for_main_server[client.client_id] = (noisy_activations, labels)
            
            # Track Laplace privacy cost if using HybridAccountant
            if isinstance(privacy_accountant, HybridAccountant) and config['dp_noise']['laplacian_sensitivity'] > 0:
                privacy_accountant.laplace_step(config['dp_noise']['epsilon_prime'])

        for client_id, (noisy_acts, lbls) in client_data_for_main_server.items():
            main_server.receive_client_data(client_id, noisy_acts, lbls)

        for client_id in client_data_for_main_server.keys():
            act_grad = main_server.forward_backward_pass(client_id)
            client_activation_grads[client_id] = act_grad

        main_server.aggregate_and_update()

        for client in clients:
            if client.client_id in client_activation_grads:
                activation_grad = client_activation_grads[client.client_id]
                noisy_wc_grad = client.local_backward_pass(activation_grad)
                client_noisy_wc_grads.append(noisy_wc_grad)
                if current_sigma > 0:
                    if isinstance(privacy_accountant, HybridAccountant):
                        privacy_accountant.gaussian_step(
                            noise_multiplier=current_sigma,
                            num_steps=1
                        )
                    else:
                        privacy_accountant.step(
                            noise_multiplier=current_sigma,
                            sampling_rate=sampling_rate,
                            num_steps=1
                        )
            else:
                print(f"Warning: No activation gradient received for Client {client.client_id}")

        for noisy_grad in client_noisy_wc_grads:
            fed_server.receive_client_update(noisy_grad)

        # Aggregate updates and update noise scale based on validation loss
        fed_server.aggregate_updates(validation_loader, main_server=main_server)
        
        round_end_time = time.time()
        print(f"--- Round {round_num + 1} finished in {round_end_time - round_start_time:.2f} seconds ---")

        if (round_num + 1) % config.get('log_interval', 10) == 0:
            if current_sigma > 0:
                epsilon, _ = privacy_accountant.get_privacy_spent(delta=target_delta)
                print(f"Round {round_num + 1}: Current Privacy Budget (ε, δ={target_delta}): ({epsilon:.4f}, {target_delta})")
                print(f"Round {round_num + 1}: Current Noise Scale (σ): {current_sigma:.4f}")
            else:
                print(f"Round {round_num + 1}: Noise disabled, privacy budget not tracked.")

    total_time = time.time() - start_time
    print(f"\nSFL Training finished in {total_time:.2f} seconds.")

    print("\nEvaluating final model...")
    final_wc_params = fed_server.get_client_model_params()
    final_ws_params = main_server.get_server_model().state_dict()

    try:
        final_global_model = get_model(config['model']).to(device)
        eval_model_client_part, eval_model_server_part = split_model(final_global_model, config['cut_layer'])
        eval_model_client_part.load_state_dict(final_wc_params)
        eval_model_server_part.load_state_dict(final_ws_params)
    except Exception as e:
        print(f"Error combining models for evaluation: {e}")
        print("Skipping evaluation.")
        final_global_model = None

    if final_global_model:
        test_accuracy = evaluate_model(final_global_model, test_loader, device)
        print(f"Final Test Accuracy: {test_accuracy:.2f}%")
    else:
        test_accuracy = None

    epsilon = None
    if current_sigma > 0:
        epsilon, final_delta = privacy_accountant.get_privacy_spent(delta=target_delta)
        print(f"Final Privacy Budget (ε, δ) for Mechanism 2 (Gaussian): ({epsilon:.4f}, {final_delta}) after {privacy_accountant.total_steps} steps.")
        print(f"Final Noise Scale (σ): {current_sigma:.4f}")
    else:
        print("Final Privacy Budget: Noise disabled for Mechanism 2.")
        
    # Save metrics to JSON file
    if out_dir:
        initial_sigma = config['dp_noise'].get('initial_sigma', 0.0)
        epsilon_prime = config['dp_noise'].get('epsilon_prime', 0.0)
        
        metrics = {
            "final_test_acc": test_accuracy / 100.0 if test_accuracy is not None else None,  # Convert to decimal
            "epsilon": epsilon,
            "delta": target_delta,
            "sigma": current_sigma,
            "sigma_init": initial_sigma,
            "epsilon_prime": epsilon_prime,
            "rounds": num_rounds,
            "epsilon_history": privacy_accountant._eps_history,
            "sigma_history": fed_server._sigma_history
        }
        
        # Add the separate epsilon values if using the hybrid accountant
        if isinstance(privacy_accountant, HybridAccountant):
            metrics["epsilon_laplace"] = privacy_accountant.epsilon_laplace
            metrics["epsilon_gaussian"] = privacy_accountant.epsilon_gaussian
            
        metrics_file = out_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main() 