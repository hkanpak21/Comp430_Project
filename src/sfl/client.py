import torch
import torch.nn as nn
import torch.optim as optim
from opacus.utils.batch_memory_manager import BatchMemoryManager
import numpy as np
import logging

# Add project root to Python path for importing dp.adaptive
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # Go up TWO levels
sys.path.insert(0, project_root)

from src.dp.adaptive import compute_trusted_client_params

class SFLClient:
    def __init__(self, client_id, model_part, dataloader, optimizer_name, lr, device, local_epochs):
        self.id = client_id
        self.model_part = model_part.to(device)
        self.dataloader = dataloader
        self.device = device
        self.local_epochs = local_epochs

        # Store optimizer name and lr for potential re-initialization with DP
        self.optimizer_name = optimizer_name
        self.lr = lr

        # Initialize optimizer (used only if not manual DP)
        self._initialize_optimizer()

        self.criterion = nn.CrossEntropyLoss() # Or appropriate loss
        self.smashed_data = None
        self.last_grad_norm = 0.0 # Store the gradient norm for feedback

    def _initialize_optimizer(self):
        """Initializes the optimizer."""
        if self.optimizer_name.lower() == 'sgd':
            self.optimizer = optim.SGD(self.model_part.parameters(), lr=self.lr)
        elif self.optimizer_name.lower() == 'adam':
            # Adam might be problematic with manual gradient updates, SGD is safer
            logging.warning("Using SGD instead of Adam for manual DP compatibility.")
            self.optimizer = optim.SGD(self.model_part.parameters(), lr=self.lr)
            # self.optimizer = optim.Adam(self.model_part.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

    def set_optimizer(self, optimizer):
        """Allows replacing the optimizer (e.g., with Opacus DP optimizer, though not used for manual DP)."""
        # This might not be needed if we stick to manual DP for adaptive_trusted_client
        self.optimizer = optimizer

    def train_step(self, server_gradients):
        """Performs one local training step (forward pass only)."""
        self.model_part.train()
        # Note: Backward pass and optimizer step happen in apply_gradients

        # Perform local epochs
        for epoch in range(self.local_epochs):
            if isinstance(self.dataloader, BatchMemoryManager):
                 # Opacus BMM handling (though Opacus optimizer won't be used for manual DP)
                for data, target in self.dataloader:
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad() # Zero grad before forward pass
                    self.smashed_data = self.model_part(data)
                    yield self.smashed_data.detach().clone().cpu(), target.cpu()
            else:
                # Standard DataLoader iteration
                for data, target in self.dataloader:
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad() # Zero grad before forward pass
                    self.smashed_data = self.model_part(data)
                    yield self.smashed_data.detach().clone().cpu(), target.cpu()

        # Return None after local epochs are done
        yield None, None

    def apply_gradients(self, gradients, dp_mode, current_sigma=None, current_C=None):
        """Applies gradients received from the server, calculates grad norm, and applies manual DP if needed."""
        if gradients is None or self.smashed_data is None:
            self.last_grad_norm = 0.0
            self.smashed_data = None
            return

        try:
            # Perform backward pass using gradients from server
            self.smashed_data.backward(gradients.to(self.device))

            # Calculate L2 norm of the *average* gradient across the batch
            # This norm is used for feedback
            total_norm_sq = 0.0
            num_params = 0
            avg_grads_list = [] # Store average grads for manual DP
            for p in self.model_part.parameters():
                if p.grad is not None:
                    # Opacus calculates per-sample grads, but here we only have the batch average grad
                    avg_grad = p.grad.detach().data # This is the gradient averaged over the batch
                    avg_grads_list.append(avg_grad)
                    param_norm_sq = avg_grad.norm(2).item() ** 2
                    total_norm_sq += param_norm_sq
                    num_params += 1
            self.last_grad_norm = (total_norm_sq ** 0.5) / num_params if num_params > 0 else 0.0 # Avg norm per parameter
            # Alternative feedback: norm of the concatenated grad vector
            # flat_avg_grad = torch.cat([g.flatten() for g in avg_grads_list])
            # self.last_grad_norm = flat_avg_grad.norm(2).item()

            # Apply DP manually if in adaptive_trusted_client mode
            if dp_mode == 'adaptive_trusted_client':
                if current_sigma is None or current_C is None:
                    logging.error("Manual DP requires sigma and C values.")
                    raise ValueError("Sigma and C must be provided for manual DP.")
                
                # 1. Calculate norm of the concatenated average gradient vector
                flat_avg_grad = torch.cat([g.flatten() for g in avg_grads_list])
                avg_grad_norm = flat_avg_grad.norm(2).item()
                
                # 2. Clip the average gradient vector
                clip_coeff = min(1.0, current_C / (avg_grad_norm + 1e-6)) # Add epsilon for stability
                
                # 3. Add noise and update parameters manually
                with torch.no_grad():
                    for i, p in enumerate(self.model_part.parameters()):
                        if p.grad is not None:
                            avg_grad = avg_grads_list[i]
                            clipped_grad = avg_grad * clip_coeff
                            
                            # Generate noise (scale by C, not sigma*C for Gaussian mechanism)
                            noise_std = current_sigma * current_C
                            noise = torch.normal(0, noise_std, size=avg_grad.shape, device=self.device)
                            
                            noised_clipped_grad = clipped_grad + noise
                            
                            # Manual optimizer step (SGD)
                            p.data -= self.lr * noised_clipped_grad
                            
                            # Zero out grad manually since optimizer.step() is bypassed
                            p.grad = None 

            else:
                # Use the standard optimizer (potentially wrapped by Opacus for other DP modes)
                self.optimizer.step()

        except RuntimeError as e:
            logging.error(f"Client {self.id}: Error during backward/step: {e}", exc_info=True)
            logging.error(f"Smashed data shape: {self.smashed_data.shape if self.smashed_data is not None else 'None'}, requires_grad: {self.smashed_data.requires_grad if self.smashed_data is not None else 'N/A'}")
            logging.error(f"Gradients shape: {gradients.shape}")
            # Zero grads manually on error if optimizer step was skipped
            if dp_mode == 'adaptive_trusted_client':
                 with torch.no_grad():
                    for p in self.model_part.parameters():
                        if p.grad is not None:
                            p.grad = None
            else:
                self.optimizer.zero_grad()
            raise e
        finally:
            # Ensure optimizer gradients are zeroed if not handled by manual update
            if dp_mode != 'adaptive_trusted_client':
                 self.optimizer.zero_grad()
            self.smashed_data = None # Clear after use

    def get_feedback_metric(self):
        """Returns the calculated feedback metric (average gradient norm)."""
        return self.last_grad_norm

    def calculate_new_dp_params(self, current_sigma, current_C, feedback_metrics, prev_avg_norm, config):
        """(Trusted Client Only) Calculates new DP parameters based on feedback."""
        if self.id != config.trusted_client_id:
            raise PermissionError("Only the trusted client can calculate new DP parameters.")
        
        logging.info(f"Trusted Client {self.id}: Calculating new DP params.")
        return compute_trusted_client_params(current_sigma, current_C, feedback_metrics, prev_avg_norm, config)

