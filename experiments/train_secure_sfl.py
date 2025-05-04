#!/usr/bin/env python3
import sys
import os
# Add the project root directory (Comp430_Project) to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
# print(f"DEBUG: Inserted into sys.path: {project_root}") # Debug print - Can be removed later
# print(f"DEBUG: Current sys.path[0]: {sys.path[0]}") # Debug print - Can be removed later

import torch
import torch.nn as nn
import numpy as np
import random
import time
import copy
from collections import OrderedDict

# Imports for flattened structure
from utils.config_parser import get_config
from datasets.data_loader import get_mnist_dataloaders, get_client_data_loaders
from models import get_model # Import from models package
from models.split_utils import split_model, get_combined_model
from dp.privacy_accountant import ManualPrivacyAccountant
from sfl.client import SFLClient
from sfl.main_server import MainServer
from sfl.fed_server import FedServer

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

def main():
    config = get_config()
    print("Configuration loaded:", config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(config['seed'])

    _, test_loader, train_dataset, _ = get_mnist_dataloaders(config)
    client_loaders = get_client_data_loaders(config, train_dataset)
    num_clients = config['num_clients']
    print(f"Loaded {config['dataset']} dataset and created {num_clients} client data loaders.")

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

    privacy_accountant = ManualPrivacyAccountant(moment_orders=None)
    dataset_size = len(train_dataset)
    batch_size = config['batch_size']
    sampling_rate = batch_size / dataset_size
    noise_multiplier = config['dp_noise']['noise_multiplier']
    clip_norm = config['dp_noise']['clip_norm']
    target_delta = float(config['dp_noise']['delta'])
    print(f"Initialized Manual Privacy Accountant. Sampling rate (q): {sampling_rate:.4f}, Noise Multiplier (z): {noise_multiplier}, Clip Norm (C): {clip_norm}, Target Delta: {target_delta}")

    start_time = time.time()
    num_rounds = config['num_rounds']
    print(f"\nStarting SFLV1 training for {num_rounds} rounds...")

    for round_num in range(num_rounds):
        round_start_time = time.time()
        print(f"\n--- Round {round_num + 1}/{num_rounds} ---")
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
                if noise_multiplier > 0:
                    privacy_accountant.step(noise_multiplier=noise_multiplier,
                                            sampling_rate=sampling_rate,
                                            num_steps=1)
            else:
                print(f"Warning: No activation gradient received for Client {client.client_id}")

        for noisy_grad in client_noisy_wc_grads:
            fed_server.receive_client_update(noisy_grad)

        fed_server.aggregate_updates()
        round_end_time = time.time()
        print(f"--- Round {round_num + 1} finished in {round_end_time - round_start_time:.2f} seconds ---")

        if (round_num + 1) % config.get('log_interval', 10) == 0:
            if noise_multiplier > 0:
                epsilon, _ = privacy_accountant.get_privacy_spent(delta=target_delta)
                print(f"Round {round_num + 1}: Current Privacy Budget (ε, δ={target_delta}): ({epsilon:.4f}, {target_delta})")
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

    if noise_multiplier > 0:
        final_epsilon, final_delta = privacy_accountant.get_privacy_spent(delta=target_delta)
        print(f"Final Privacy Budget (ε, δ) for Mechanism 2 (Gaussian): ({final_epsilon:.4f}, {final_delta}) after {privacy_accountant.total_steps} steps.")
    else:
        print("Final Privacy Budget: Noise disabled for Mechanism 2.")

if __name__ == "__main__":
    main() 