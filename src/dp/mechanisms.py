import torch
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
from opacus.optimizers import DPOptimizer
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch.utils.data import DataLoader
import logging

def attach_dp_mechanism(args, model, optimizer, dataloader):
    """Attaches the appropriate DP mechanism using Opacus based on args."""
    privacy_engine = None
    accountant = None
    dp_optimizer = None
    dp_dataloader = None

    if args.dp_mode == 'none':
        logging.info("DP Mode: None. Running standard SFL.")
        # Return original objects
        dp_optimizer = optimizer # No DP optimizer wrapper needed
        dp_dataloader = dataloader # No DP dataloader needed

    elif args.dp_mode == 'vanilla':
        logging.info(f"DP Mode: Vanilla Gaussian. Target Epsilon: {args.target_epsilon}, Target Delta: {args.target_delta}, Noise Multiplier: {args.noise_multiplier}, Max Grad Norm: {args.max_grad_norm}")
        privacy_engine = PrivacyEngine(accountant='rdp') # Use RDP accountant by default

        # Note: In SFL, DP is typically applied either to client-side gradients before sending smashed data gradients,
        # or to the smashed data gradients received by the server before aggregation, or on the server update itself.
        # Applying it on the client-side optimizer *before* the split is the most common simulation approach inspired by standard DP-FedAvg.
        # This means we wrap the client's optimizer and model part.

        # Opacus modifies the model in-place (e.g., replacing BatchNorm)
        # It also wraps the optimizer.
        # The dataloader might need wrapping for batch handling (e.g., Poisson sampling).

        try:
            # Attach Opacus to the client model part and optimizer
            # IMPORTANT: Opacus usually expects the *entire* model. In SFL, we apply DP
            # to the client part. This might require careful consideration of sensitivity
            # and privacy accounting, as only part of the gradient computation is protected locally.
            # For this simulation, we apply it directly to the client part.
            model, dp_optimizer, dp_dataloader = privacy_engine.make_private(
                module=model, # Client model part
                optimizer=optimizer,
                data_loader=dataloader,
                noise_multiplier=args.noise_multiplier,
                max_grad_norm=args.max_grad_norm,
                poisson_sampling=True, # Use Poisson sampling for better composition
            )
            accountant = privacy_engine.accountant
            logging.info(f"Attached Opacus PrivacyEngine (Vanilla) to client model {type(model).__name__}")

            # If the original dataloader was simple, Opacus might replace it
            # If it was already a BatchMemoryManager, Opacus might wrap it further or handle it internally.
            # We need to ensure the training loop correctly handles the potentially modified dataloader.

        except Exception as e:
            logging.error(f"Error attaching Opacus PrivacyEngine: {e}")
            # Fallback or re-raise
            raise e

    # --- Placeholder for Adaptive DP Modes --- #
    elif args.dp_mode in ['adaptive_clipping', 'adaptive_noise', 'awdp']:
        # Adaptive DP is not directly supported by vanilla Opacus PrivacyEngine `make_private`.
        # We need to implement custom logic, potentially using Opacus components like
        # DPOptimizer and manually managing noise/clipping.
        logging.warning(f"DP Mode: {args.dp_mode}. Custom implementation required. Using Vanilla DP setup for now.")

        # For now, fall back to vanilla setup for structure, but real adaptivity needs more work.
        privacy_engine = PrivacyEngine(accountant='rdp')
        try:
            model, dp_optimizer, dp_dataloader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=dataloader,
                noise_multiplier=args.noise_multiplier, # Initial noise?
                max_grad_norm=args.max_grad_norm,     # Initial clipping?
                poisson_sampling=True,
            )
            accountant = privacy_engine.accountant
            logging.info(f"Attached Opacus PrivacyEngine (Placeholder for {args.dp_mode}) to client model {type(model).__name__}")
        except Exception as e:
             logging.error(f"Error attaching Opacus PrivacyEngine for adaptive placeholder: {e}")
             raise e
        # --- End Placeholder --- #

    else:
        raise ValueError(f"Unsupported DP mode: {args.dp_mode}")

    # Return the potentially modified model, optimizer, dataloader, and the accountant
    return model, dp_optimizer, dp_dataloader, accountant


def get_privacy_spent(accountant, delta):
    """Gets the current epsilon spent from the accountant."""
    if accountant is None:
        return 0.0 # No DP, epsilon is 0 (or infinite, but 0 makes more sense here)
    try:
        # Use the get_epsilon method which is standard across Opacus accountants
        return accountant.get_epsilon(delta=delta)
    except Exception as e:
        logging.error(f"Error getting privacy spent: {e}")
        return float('inf') # Indicate an error or undefined budget

# Note: Actual adaptive logic (adjusting noise/clipping dynamically)
# will need to be implemented in the training loop or custom DP optimizers/hooks.
# The `adaptive.py` file is intended for this logic. 