import torch
import torch.nn as nn
import argparse
import logging
import time
from copy import deepcopy
import numpy as np

from src.datasets.mnist import get_mnist_datasets, split_mnist_data, get_client_dataloader, get_test_dataloader
from src.models.simple_dnn import SimpleDNN, split_model
from src.sfl.client import Client
from src.sfl.main_server import MainServer
from src.sfl.fed_server import FedServer

# Basic Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate(fed_server, main_server, test_loader, device):
    """Evaluates the global model (combined client and server parts) on the test set."""
    client_model = fed_server.get_global_model().to(device)
    server_model = main_server.get_server_model().to(device)
    client_model.eval()
    server_model.eval()

    correct = 0
    total = 0
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Forward pass through client model
            smashed_data = client_model(data)

            # Forward pass through server model
            outputs = server_model(smashed_data)

            loss = criterion(outputs, target)
            test_loss += loss.item() * data.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    avg_test_loss = test_loss / total if total > 0 else 0.0
    logging.info(f"Evaluation - Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return accuracy, avg_test_loss

def main(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- 1. Data Loading and Splitting ---
    logging.info("Loading MNIST dataset...")
    train_dataset, test_dataset = get_mnist_datasets(data_path=args.data_path)
    logging.info("Splitting data among clients...")
    client_datasets, client_distributions = split_mnist_data(
        train_dataset,
        args.num_clients,
        split_mode=args.data_split_mode,
        non_iid_alpha=args.non_iid_alpha,
        seed=args.seed
    )
    client_dataloaders = [
        get_client_dataloader(ds, args.batch_size) for ds in client_datasets
    ]
    test_loader = get_test_dataloader(test_dataset, args.batch_size * 2) # Use larger batch for testing

    # --- 2. Model Definition and Splitting ---
    logging.info("Creating and splitting the model...")
    # Example: SimpleDNN with 2 hidden layers
    full_model = SimpleDNN(hidden_layers=[128, 64], num_classes=10)
    # Note: split_model expects the model to have a `.network` attribute which is the Sequential part
    initial_client_model_part, initial_server_model_part = split_model(full_model, args.cut_layer)

    # --- 3. Initialization ---
    logging.info("Initializing Servers and Clients...")
    fed_server = FedServer(initial_client_model=initial_client_model_part)
    main_server = MainServer(
        server_model_part=initial_server_model_part,
        device=device,
        optimizer_name=args.optimizer,
        lr=args.lr_server
    )
    clients = []
    for i in range(args.num_clients):
        client = Client(
            client_id=i,
            initial_model_part=initial_client_model_part, # Initial model is same for all
            dataloader=client_dataloaders[i],
            optimizer_name=args.optimizer,
            lr=args.lr_client,
            device=device,
            local_epochs=args.local_epochs
        )
        clients.append(client)

    # --- 4. Training Loop ---
    logging.info("Starting SplitFed Training...")
    start_time = time.time()

    for round_num in range(args.rounds):
        round_start_time = time.time()
        logging.info(f"--- Round {round_num + 1}/{args.rounds} ---")

        # --- FedServer: Distribute global client model ---
        global_client_state = fed_server.get_global_model_state()
        active_clients_in_round = clients # For now, assume all clients participate
        # In future, implement client selection (e.g., random subset)
        logging.info(f" Participating clients: {[c.id for c in active_clients_in_round]}")

        for client in active_clients_in_round:
             client.set_model(deepcopy(global_client_state))
             client.reset_training()

        # --- Client Training & MainServer Interaction ---
        # This loop simulates the back-and-forth for batches within the round
        # We collect all client forward passes first, then server processes, then clients backward
        # (More realistic might interleave, but this is simpler simulation)

        client_outputs_batches = {client.id: [] for client in active_clients_in_round} # Store (smashed_data, target, batch_idx)
        clients_done_this_round = {client.id: False for client in active_clients_in_round}
        num_clients_done = 0
        total_batches_processed = 0

        while num_clients_done < len(active_clients_in_round):
            batches_for_server_step = [] # Collect (client_id, smashed_data, target, batch_idx)
            
            # Clients perform one local forward step
            for client in active_clients_in_round:
                 if not clients_done_this_round[client.id]:
                     smashed_data, target, batch_idx, is_done = client.local_step()
                     if is_done:
                         if not clients_done_this_round[client.id]: # Mark done only once
                              logging.debug(f"Client {client.id} finished local training for round {round_num+1}.")
                              clients_done_this_round[client.id] = True
                              num_clients_done += 1
                     elif smashed_data is not None:
                         batches_for_server_step.append((client.id, smashed_data, target, batch_idx))
                         total_batches_processed += 1

            if not batches_for_server_step: # Break if no clients produced data (all done)
                if num_clients_done == len(active_clients_in_round):
                     break # All clients are confirmed done
                else:
                     continue # Some might still be processing epochs

            # --- MainServer: Process collected batch data ---
            # Group data for MainServer processing (simple approach: one server step per client step collect)
            smashed_data_list = [b[1] for b in batches_for_server_step]
            targets_list = [b[2] for b in batches_for_server_step]
            
            if not smashed_data_list: # Should not happen if loop condition is correct
                 logging.warning("No smashed data collected for server step, but not all clients done?")
                 continue
                 
            server_loss, gradients_list = main_server.process_batch(smashed_data_list, targets_list)
            # logging.debug(f" MainServer processed {len(smashed_data_list)} batches. Loss: {server_loss:.4f}")

            # --- Clients: Apply gradients ---
            if len(gradients_list) != len(batches_for_server_step):
                 logging.error(f"Mismatch between gradients ({len(gradients_list)}) and client batches ({len(batches_for_server_step)})! Skipping gradient application.")
            else:
                for i, grad in enumerate(gradients_list):
                    client_id, _, _, batch_idx = batches_for_server_step[i]
                    clients[client_id].apply_gradients(grad, batch_idx)

        # --- FedServer: Aggregate client models ---
        client_states_to_aggregate = []
        client_samples_to_aggregate = []
        for client in active_clients_in_round:
            client_states_to_aggregate.append(client.get_model_state())
            client_samples_to_aggregate.append(client.get_sample_count())
        
        fed_server.aggregate_models(client_states_to_aggregate, client_samples_to_aggregate)

        # --- Evaluation ---
        if (round_num + 1) % args.eval_every == 0:
            evaluate(fed_server, main_server, test_loader, device)
            
        round_duration = time.time() - round_start_time
        logging.info(f"Round {round_num + 1} finished in {round_duration:.2f}s. Total batches processed: {total_batches_processed}")


    total_time = time.time() - start_time
    logging.info(f"--- Training Finished --- Total Time: {total_time:.2f}s")

    # Final evaluation
    logging.info("Performing final evaluation...")
    evaluate(fed_server, main_server, test_loader, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split Federated Learning Simulation')
    
    # General arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to download/load MNIST data')
    
    # Training arguments
    parser.add_argument('--rounds', type=int, default=10, help='Number of communication rounds')
    parser.add_argument('--num_clients', type=int, default=5, help='Number of clients')
    parser.add_argument('--local_epochs', type=int, default=1, help='Number of local epochs for each client per round')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and testing')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'], help='Optimizer type')
    parser.add_argument('--lr_client', type=float, default=0.01, help='Learning rate for client optimizer')
    parser.add_argument('--lr_server', type=float, default=0.01, help='Learning rate for main server optimizer')
    parser.add_argument('--eval_every', type=int, default=1, help='Evaluate model every N rounds')

    # Data distribution arguments
    parser.add_argument('--data_split_mode', type=str, default='iid', choices=['iid', 'non-iid'], help='Data distribution mode')
    parser.add_argument('--non_iid_alpha', type=float, default=0.1, help='Alpha parameter for Non-IID Dirichlet distribution (smaller is more non-iid)')

    # Model splitting arguments
    parser.add_argument('--cut_layer', type=int, default=2, help='Index of the layer *after* which to split the model (client gets layers <= index)')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    main(args) 