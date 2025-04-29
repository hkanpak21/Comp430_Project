import torch
import numpy as np
import random
import os
import logging
from tqdm import tqdm
from copy import deepcopy

# Add project root to Python path
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # Adjust based on train_sfl_dp.py location
sys.path.insert(0, project_root)

from src.utils.config import parse_args
from src.utils.logger import setup_logger, log_results
from src.datasets.data_loader import get_datasets, get_data_loaders
from src.models.base_model import get_model
from src.sfl.client import SFLClient
from src.sfl.server import SFLServer
from src.dp.mechanisms import attach_dp_mechanism, get_privacy_spent
from src.dp.adaptive import (
    update_clipping_norm_adaptive, # Placeholder
    update_noise_multiplier_adaptive, # Placeholder
    apply_awdp, # Placeholder
    apply_priority_noise # Placeholder
)

def main(args):
    # --- Setup --- # 
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        # Potentially add deterministic CuDNN settings, but they can impact performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    # Setup Logger
    logger, log_filepath = setup_logger(args.log_dir, args.log_filename, args)
    logger.info("Starting SFL with Adaptive DP Experiment")
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"Log file: {log_filepath}")
    logger.info(f"Using device: {args.device}")

    # --- Data --- #
    logger.info(f"Loading dataset: {args.dataset} ({args.data_distribution} distribution)")
    train_dataset, test_dataset = get_datasets(args.dataset, args.data_root)
    client_loaders, test_loader = get_data_loaders(args, train_dataset, test_dataset)
    num_classes = len(train_dataset.classes) if hasattr(train_dataset, 'classes') else np.unique(train_dataset.tensors[1].cpu().numpy()).shape[0]
    logger.info(f"Data loaded. Num clients: {args.num_clients}, Num test samples: {len(test_dataset)}")

    # --- Model --- #
    logger.info(f"Creating model: {args.model}, Splitting at layer: {args.split_layer}")
    # Create one instance of the *full* model first to easily get parts
    full_model = get_model(args.model, num_classes, args.dataset, args.split_layer) # This also performs the split internally
    client_model_part_template = full_model.get_client_part()
    server_model_part_template = full_model.get_server_part()
    logger.info("Model parts created.")
    # logger.info(f"Client Model Part:\\n{client_model_part_template}")
    # logger.info(f"Server Model Part:\\n{server_model_part_template}")

    # --- SFL Setup --- #
    server = SFLServer(
        model_part=deepcopy(server_model_part_template), # Server gets its own copy
        test_loader=test_loader,
        optimizer_name=args.optimizer,
        lr=args.lr,
        device=args.device
    )

    clients = []
    client_accountants = {} # To store DP accountants for each client if DP is enabled
    for i in range(args.num_clients):
        client_model_part = deepcopy(client_model_part_template) # Each client gets a copy
        client_optimizer_class = getattr(torch.optim, args.optimizer)
        client_optimizer = client_optimizer_class(client_model_part.parameters(), lr=args.lr)
        client_dataloader = client_loaders[i]
        accountant = None # Default

        # Attach DP if enabled
        if args.dp_mode != 'none':
            try:
                # Pass the client's specific model part, optimizer, and dataloader
                client_model_part, client_optimizer, client_dataloader, accountant = attach_dp_mechanism(
                    args, client_model_part, client_optimizer, client_dataloader
                )
                if accountant:
                    client_accountants[i] = accountant
                logger.info(f"DP mechanism ({args.dp_mode}) attached for client {i}")
            except Exception as e:
                logger.error(f"Failed to attach DP for client {i}: {e}. Exiting.")
                return

        clients.append(SFLClient(
            client_id=i,
            model_part=client_model_part,
            dataloader=client_dataloader,
            optimizer_name=args.optimizer, # Pass original name, optimizer object is already created/wrapped
            lr=args.lr,
            device=args.device,
            local_epochs=args.local_epochs
        ))

    logger.info(f"Initialized {len(clients)} clients and server.")

    # --- Training Loop --- #
    logger.info(f"Starting training for {args.epochs} rounds.")
    global_step = 0
    client_iterators = {} # Store data iterators for clients
    client_smashed_data_cache = {} # Store smashed data from clients
    client_targets_cache = {} # Store targets corresponding to smashed data
    client_gradients_cache = {} # Store gradients received from server

    for epoch in range(args.epochs):
        logger.info(f"--- Round {epoch + 1}/{args.epochs} ---")
        round_loss = 0.0
        num_batches_processed_round = 0

        # Select clients for the round
        if args.clients_per_round == args.num_clients:
            selected_client_indices = list(range(args.num_clients))
        else:
            selected_client_indices = random.sample(range(args.num_clients), args.clients_per_round)
        logger.info(f"Selected clients for round {epoch + 1}: {selected_client_indices}")

        # --- Client-Side Training (Simulated Batch-wise) --- # 
        # In a real system, this would be asynchronous or parallel.
        # Here, we simulate step-by-step interaction for clarity.

        active_clients = {idx: True for idx in selected_client_indices} # Track clients still processing data in this round
        server_batch_smashed = []
        server_batch_targets = []
        client_map_for_server_batch = [] # Track which client corresponds to each part of server batch

        # Initialize or reset iterators for selected clients
        for client_idx in selected_client_indices:
            if client_idx not in client_iterators:
                # Pass previous round's gradients if available
                prev_grads = client_gradients_cache.get(client_idx, None)
                client_iterators[client_idx] = clients[client_idx].train_step(prev_grads)
                client_gradients_cache[client_idx] = None # Clear gradients after use

        # Loop until all selected clients have finished their local epochs for this round
        while any(active_clients.values()):
            server_batch_smashed.clear()
            server_batch_targets.clear()
            client_map_for_server_batch.clear()

            # Collect one batch of smashed data from each active client
            for client_idx in selected_client_indices:
                if active_clients[client_idx]:
                    try:
                        smashed_data, targets = next(client_iterators[client_idx])

                        if smashed_data is not None:
                            # Store for server processing
                            server_batch_smashed.append(smashed_data)
                            server_batch_targets.append(targets)
                            client_map_for_server_batch.append(client_idx)
                            # Cache the latest smashed data for the client (needed for backward pass)
                            client_smashed_data_cache[client_idx] = smashed_data
                        else:
                            # Client finished local epochs for this round
                            active_clients[client_idx] = False
                            del client_iterators[client_idx] # Remove iterator for this round
                            logger.debug(f"Client {client_idx} finished local epochs for round {epoch + 1}.")

                    except StopIteration:
                        # Should not happen with yield-based approach, but handle defensively
                        active_clients[client_idx] = False
                        if client_idx in client_iterators: del client_iterators[client_idx]
                        logger.debug(f"Client {client_idx} iterator stopped for round {epoch + 1}.")
                    except Exception as e:
                         logger.error(f"Error during client {client_idx} train_step: {e}", exc_info=True)
                         active_clients[client_idx] = False # Stop processing for this client
                         if client_idx in client_iterators: del client_iterators[client_idx]


            # If we collected any smashed data, process it on the server
            if server_batch_smashed:
                # --- Server-Side Processing --- #
                server_loss, gradients_list = server.train_step(server_batch_smashed, server_batch_targets)
                round_loss += server_loss * sum(s.size(0) for s in server_batch_smashed) # Weighted loss
                num_batches_processed_round += 1
                global_step += 1

                # --- Send Gradients Back to Clients --- #
                if len(gradients_list) != len(client_map_for_server_batch):
                     logger.error(f"Mismatch between gradients ({len(gradients_list)}) and client map ({len(client_map_for_server_batch)})!")
                     # Handle error appropriately - skip gradient update? 
                else:
                    for i, client_idx in enumerate(client_map_for_server_batch):
                        # Apply gradients on the client side
                        # Requires the cached smashed data from the forward pass
                        if client_idx in client_smashed_data_cache:
                            cached_smashed = client_smashed_data_cache[client_idx]
                             # Re-attach grad_fn if lost during CPU transfer (shouldn't happen with detach/clone)
                             # if not cached_smashed.requires_grad:
                             #     cached_smashed.requires_grad_(True) # This might be tricky

                            # The client optimizer step is triggered *by* the backward pass using server gradients
                            # Store gradients for the *next* step/batch if local_epochs > 1, or apply now?
                            # Current SFLClient design applies grads from *previous* server step at the *start* of train_step.
                            # So, cache the received gradients.
                            client_gradients_cache[client_idx] = gradients_list[i]
                            # clients[client_idx].apply_gradients(gradients_list[i])
                            del client_smashed_data_cache[client_idx] # Clear cache after potential use
                        else:
                            logger.warning(f"Client {client_idx} has no cached smashed data for gradient application.")

            # Apply adaptive DP updates here if needed (e.g., per batch or per server step)
            if args.dp_mode == 'adaptive_clipping':
                # Needs access to per-sample grads *before* server step aggregation/optimizer
                # This is complex in SFL. Placeholder call.
                # update_clipping_norm_adaptive(server.optimizer, args.adaptive_clipping_quantile) # Example on server?
                pass
            elif args.dp_mode == 'awdp':
                # apply_awdp(...) # Placeholder
                pass

        # --- End of Round --- # 
        avg_round_loss = round_loss / len(train_dataset) if len(train_dataset) > 0 else 0.0 # Approximate loss over dataset

        # Apply adaptive DP updates here (e.g., per round)
        if args.dp_mode == 'adaptive_noise':
             # Placeholder call - needs access to client optimizers if applied per client
             # for client_idx in selected_client_indices:
             #    client_opt = clients[client_idx].optimizer # Need access to the actual (potentially wrapped) optimizer
             #    update_noise_multiplier_adaptive(client_opt, epoch, args.epochs, args.adaptive_noise_decay, args.noise_multiplier)
             pass

        # Calculate privacy budget spent this round (approximate)
        epsilon_per_round = 0.0
        total_epsilon = 0.0
        if args.dp_mode != 'none':
            # Get budget from the first client's accountant (assuming identical params)
            # A more robust approach might average or track per client.
            if client_accountants:
                 first_accountant = next(iter(client_accountants.values()))
                 try:
                     # Calculate epsilon spent in this round approximately
                     # This is tricky as Opacus tracks total budget. We might need to store
                     # the previous epsilon and subtract.
                     current_total_epsilon = get_privacy_spent(first_accountant, args.target_delta)
                     # Store previous value to calculate per-round budget next time?
                     # For simplicity, just log the total spent so far.
                     total_epsilon = current_total_epsilon
                     epsilon_per_round = total_epsilon / (epoch + 1) # Crude average per round

                 except Exception as e:
                     logger.error(f"Could not calculate privacy budget: {e}")
                     total_epsilon = float('inf')
            else:
                logger.warning("DP mode enabled, but no accountants found.")


        # Evaluate the global model (using one client's final model part)
        # Note: Client models might diverge slightly. Using client 0 as representative.
        # A better approach might average client models (if feasible/secure).
        accuracy, test_loss = server.evaluate(clients[0].model_part)

        logger.info(f"Round {epoch + 1} finished. Avg Train Loss: {avg_round_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
        if args.dp_mode != 'none':
            logger.info(f"Privacy Budget: Approx. Total Epsilon = {total_epsilon:.4f} (Delta = {args.target_delta})")

        # Log results
        log_results(log_filepath, args, epoch + 1, avg_round_loss, accuracy, epsilon_per_round, total_epsilon)

        # Check for privacy budget exhaustion
        if args.dp_mode != 'none' and total_epsilon > args.target_epsilon:
            logger.warning(f"Target epsilon {args.target_epsilon} exceeded at epoch {epoch + 1}. Stopping training.")
            break

    logger.info("Training finished.")
    # Final evaluation
    final_accuracy, final_test_loss = server.evaluate(clients[0].model_part)
    final_epsilon = 0.0
    if args.dp_mode != 'none' and client_accountants:
         first_accountant = next(iter(client_accountants.values()))
         final_epsilon = get_privacy_spent(first_accountant, args.target_delta)

    logger.info(f"Final Results - Test Loss: {final_test_loss:.4f}, Test Accuracy: {final_accuracy:.2f}%" + (f", Total Epsilon: {final_epsilon:.4f}" if args.dp_mode != 'none' else ""))

if __name__ == "__main__":
    args = parse_args()
    main(args) 