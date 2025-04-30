import torch
import numpy as np
import random
import os
import logging
from tqdm import tqdm
from copy import deepcopy

# Add project root to Python path
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Go up ONE level
sys.path.insert(0, project_root)

from src.utils.config import parse_args
from src.utils.logger import setup_logger, log_results
from src.datasets.data_loader import get_datasets, get_data_loaders
from src.models.base_model import get_model
from src.sfl.client import SFLClient
from src.sfl.server import SFLServer
from src.dp.mechanisms import attach_dp_mechanism, get_privacy_spent
from src.dp.adaptive import (
    compute_trusted_client_params, # Import the new function
    update_clipping_norm_adaptive, # Placeholder
    update_noise_multiplier_adaptive, # Placeholder
    apply_awdp, # Placeholder
    apply_priority_noise # Placeholder
)
from opacus.optimizers import DPOptimizer # Import DPOptimizer to check type

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
        # Initialize the base optimizer first
        base_optimizer = client_optimizer_class(client_model_part.parameters(), lr=args.lr)
        client_dataloader = client_loaders[i]
        accountant = None # Default

        # Initialize client first
        client = SFLClient(
            client_id=i,
            model_part=client_model_part,
            dataloader=client_dataloader, # Will be potentially replaced by DP dataloader
            optimizer_name=args.optimizer, # Pass original name
            lr=args.lr,
            device=args.device,
            local_epochs=args.local_epochs
        )
        # Set the base optimizer initially
        client.set_optimizer(base_optimizer)

        # Attach DP if enabled (or prepare for manual DP)
        if args.dp_mode != 'none':
            try:
                # Pass the client's model part, base optimizer, and dataloader
                # For manual DP ('adaptive_trusted_client'), this returns original objects
                dp_model_part, dp_optimizer, dp_dataloader, accountant = attach_dp_mechanism(
                    args, client.model_part, client.optimizer, client.dataloader
                )
                # Update client's attributes (only relevant if Opacus was used)
                client.model_part = dp_model_part
                client.set_optimizer(dp_optimizer) # Replace optimizer if DP-wrapped
                client.dataloader = dp_dataloader # Replace dataloader if modified

                if accountant: # Only if Opacus accountant was returned
                    client_accountants[i] = accountant
                logger.info(f"DP mechanism ({args.dp_mode}) setup for client {i}")
            except Exception as e:
                logger.error(f"Failed to setup DP for client {i}: {e}. Exiting.", exc_info=True)
                return

        clients.append(client)

    logger.info(f"Initialized {len(clients)} clients and server.")

    # --- Training Loop --- #
    logger.info(f"Starting training for {args.epochs} rounds.")
    global_step = 0
    client_iterators = {} # Store data iterators for clients
    # client_smashed_data_cache = {} # No longer needed, client stores its own
    # client_targets_cache = {} # No longer needed
    # client_gradients_cache = {} # No longer needed, apply immediately
    client_feedback_cache = {} # Store feedback metrics from clients for the current round

    # DP parameters for adaptive_trusted_client mode
    current_sigma = args.noise_multiplier
    current_C = args.max_grad_norm
    prev_avg_norm = None # Store the average norm from the previous round
    sigma_history = []
    C_history = []

    for epoch in range(args.epochs):
        logger.info(f"--- Round {epoch + 1}/{args.epochs} ---")
        if args.dp_mode == 'adaptive_trusted_client':
             logger.info(f"Round {epoch + 1} using sigma={current_sigma:.4f}, C={current_C:.4f}")
             sigma_history.append(current_sigma)
             C_history.append(current_C)
             # No need to update Opacus optimizers anymore for manual DP

        round_loss = 0.0
        num_batches_processed_round = 0
        client_feedback_cache.clear() # Clear feedback cache for the new round

        # Select clients for the round
        if args.clients_per_round == args.num_clients:
            selected_client_indices = list(range(args.num_clients))
        else:
            selected_client_indices = random.sample(range(args.num_clients), args.clients_per_round)
        logger.info(f"Selected clients for round {epoch + 1}: {selected_client_indices}")

        # --- Client-Side Training (Simulated Batch-wise) --- # 
        active_clients = {idx: True for idx in selected_client_indices} # Track clients still processing data in this round
        server_batch_smashed = []
        server_batch_targets = []
        client_map_for_server_batch = [] # Track which client corresponds to each part of server batch

        # Initialize or reset iterators for selected clients
        for client_idx in selected_client_indices:
            if client_idx not in client_iterators:
                # Pass None for server_gradients as apply_gradients handles it now
                client_iterators[client_idx] = clients[client_idx].train_step(None)

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
                            # Client stores its own smashed_data internally now
                        else:
                            # Client finished local epochs for this round
                            active_clients[client_idx] = False
                            del client_iterators[client_idx] # Remove iterator for this round
                            logger.debug(f"Client {client_idx} finished local epochs for round {epoch + 1}.")

                    except StopIteration:
                        active_clients[client_idx] = False
                        if client_idx in client_iterators: del client_iterators[client_idx]
                        logger.debug(f"Client {client_idx} iterator stopped for round {epoch + 1}.")
                    except Exception as e:
                         logger.error(f"Error during client {client_idx} train_step: {e}", exc_info=True)
                         active_clients[client_idx] = False
                         if client_idx in client_iterators: del client_iterators[client_idx]

            # If we collected any smashed data, process it on the server
            if server_batch_smashed:
                # --- Server-Side Processing --- #
                server_loss, gradients_list = server.train_step(server_batch_smashed, server_batch_targets)
                round_loss += server_loss * sum(s.size(0) for s in server_batch_smashed) # Weighted loss
                num_batches_processed_round += 1
                global_step += 1

                # --- Send Gradients Back to Clients & Collect Feedback --- #
                if len(gradients_list) != len(client_map_for_server_batch):
                     logger.error(f"Mismatch between gradients ({len(gradients_list)}) and client map ({len(client_map_for_server_batch)})!")
                else:
                    for i, client_idx in enumerate(client_map_for_server_batch):
                        client = clients[client_idx]
                        # Apply gradients on the client side (triggers backward, optional manual DP, and optimizer step)
                        client.apply_gradients(gradients_list[i], args.dp_mode, current_sigma, current_C)
                        # Collect feedback metric *after* gradients are applied and norm is calculated
                        feedback = client.get_feedback_metric()
                        if client_idx not in client_feedback_cache:
                            client_feedback_cache[client_idx] = []
                        client_feedback_cache[client_idx].append(feedback)
                        # logger.debug(f"Client {client_idx} feedback (grad_norm): {feedback:.4f}")

            # Adaptive DP updates (other modes - placeholders)
            if args.dp_mode == 'adaptive_clipping':
                pass
            elif args.dp_mode == 'awdp':
                pass

        # --- End of Round --- # 
        # Calculate approximate average training loss for the round
        num_train_samples = sum(len(clients[idx].dataloader.dataset) for idx in selected_client_indices)
        avg_round_loss = round_loss / num_train_samples if num_train_samples > 0 else 0.0

        # --- Adaptive Trusted Client DP Update --- #
        if args.dp_mode == 'adaptive_trusted_client':
            # Aggregate feedback metrics for the round (e.g., average grad norm per client)
            round_feedback_metrics = []
            for client_idx in selected_client_indices:
                if client_idx in client_feedback_cache and client_feedback_cache[client_idx]:
                    avg_client_feedback = np.mean(client_feedback_cache[client_idx])
                    round_feedback_metrics.append(avg_client_feedback)
                else:
                    logger.warning(f"No feedback metrics found for client {client_idx} in round {epoch + 1}")
            
            if round_feedback_metrics:
                # Get the trusted client
                trusted_client = clients[args.trusted_client_id]
                try:
                    # Calculate new parameters using the trusted client's method
                    new_sigma, new_C, current_avg_norm = trusted_client.calculate_new_dp_params(
                        current_sigma, current_C, round_feedback_metrics, prev_avg_norm, args
                    )
                    # Update parameters for the *next* round
                    current_sigma = new_sigma
                    current_C = new_C
                    prev_avg_norm = current_avg_norm # Store the calculated average for the next comparison
                except Exception as e:
                    logger.error(f"Error calculating new DP parameters via trusted client: {e}", exc_info=True)
            else:
                logger.warning(f"No feedback metrics collected in round {epoch + 1}. DP parameters remain unchanged.")

        # Calculate privacy budget spent (Manual DP - Approximation Needed)
        # For manual DP, we need a separate way to track privacy budget.
        # This is complex as sigma changes. A simple approximation might use the *average* sigma over the rounds,
        # but this is not rigorous. For now, we report 0 as Opacus accountant isn't used.
        total_epsilon = 0.0
        epsilon_per_round = 0.0
        if args.dp_mode == 'adaptive_trusted_client':
            # TODO: Implement manual privacy accounting if needed (e.g., using RDP accountants manually)
            logger.warning("Manual DP mode: Privacy budget calculation is approximate/not implemented.")
            # Example approximation (very rough): Calculate based on average sigma so far
            # avg_sigma = np.mean(sigma_history) if sigma_history else args.noise_multiplier
            # steps = num_batches_processed_round * (epoch + 1) # Rough total steps
            # sample_rate = args.batch_size / len(train_dataset) # Needs adjustment for client sampling
            # approx_epsilon = compute_epsilon(steps, avg_sigma, sample_rate, args.target_delta)
            pass
        elif args.dp_mode != 'none': # For other DP modes using Opacus
            if client_accountants:
                 first_accountant = next(iter(client_accountants.values()))
                 try:
                     total_epsilon = get_privacy_spent(first_accountant, args.target_delta)
                 except Exception as e:
                     logger.error(f"Could not calculate privacy budget: {e}")
                     total_epsilon = float('inf')
            else:
                logger.warning("DP mode enabled, but no accountants found.")
            epsilon_per_round = total_epsilon / (epoch + 1) if epoch >= 0 else 0.0

        # Evaluate the global model (using one client's final model part)
        accuracy, test_loss = server.evaluate(clients[0].model_part)

        logger.info(f"Round {epoch + 1} finished. Avg Train Loss: {avg_round_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
        if args.dp_mode != 'none':
            logger.info(f"Privacy Budget: Approx Total Epsilon = {total_epsilon:.4f} (Delta = {args.target_delta})")
            if args.dp_mode == 'adaptive_trusted_client':
                 logger.info(f"Adaptive Params History (Sigma): {sigma_history}")
                 logger.info(f"Adaptive Params History (C): {C_history}")

        # Log results
        log_results(log_filepath, args, epoch + 1, avg_round_loss, accuracy, epsilon_per_round, total_epsilon, current_sigma, current_C)

        # Check for privacy budget exhaustion (only if using Opacus accountant)
        if args.dp_mode != 'none' and args.dp_mode != 'adaptive_trusted_client' and total_epsilon > args.target_epsilon:
            logger.warning(f"Target epsilon {args.target_epsilon} exceeded at epoch {epoch + 1}. Stopping training.")
            break

    logger.info("Training finished.")
    # Final evaluation
    final_accuracy, final_test_loss = server.evaluate(clients[0].model_part)
    final_epsilon = 0.0
    if args.dp_mode != 'none' and args.dp_mode != 'adaptive_trusted_client' and client_accountants:
         first_accountant = next(iter(client_accountants.values()))
         final_epsilon = get_privacy_spent(first_accountant, args.target_delta)

    logger.info(f"Final Results - Test Loss: {final_test_loss:.4f}, Test Accuracy: {final_accuracy:.2f}%" + (f", Approx Total Epsilon: {final_epsilon:.4f}" if args.dp_mode != 'none' else ""))
    if args.dp_mode == 'adaptive_trusted_client':
        logger.info(f"Final Sigma: {current_sigma:.4f}, Final C: {current_C:.4f}")
        logger.info(f"Sigma History: {sigma_history}")
        logger.info(f"C History: {C_history}")

if __name__ == "__main__":
    args = parse_args()
    main(args)

