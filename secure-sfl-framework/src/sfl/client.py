import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import OrderedDict
from typing import Tuple, Dict

# Assuming noise_utils.py is in src/dp/
from ..dp import noise_utils

class SFLClient:
    """
    Client in SFLV1.
    Manages local data, client-side model (WC), performs local computations,
    applies noise, and communicates intermediate results.
    """
    def __init__(self, client_id: int, client_model: nn.Module, dataloader: DataLoader, config: dict, device: torch.device):
        """
        Args:
            client_id: Unique identifier for the client.
            client_model: A *copy* of the initial client-side model (WC) architecture.
            dataloader: DataLoader for the client's local dataset partition.
            config: Configuration dictionary.
            device: The torch device ('cpu' or 'cuda').
        """
        self.client_id = client_id
        self.client_model = client_model.to(device)
        self.dataloader = dataloader
        self.config = config
        self.device = device
        self.optimizer = self._create_optimizer() # Optimizer for WC

        # Store intermediate activation for backward pass
        self._activations = None
        self._data_batch = None # Store data batch to access individual samples for clipping
        self._labels_batch = None # Store labels corresponding to the data batch

    def _create_optimizer(self) -> optim.Optimizer:
        """Creates the optimizer for the client-side model (WC)."""
        lr = self.config.get('lr', 0.01)
        optimizer_name = self.config.get('optimizer', 'SGD').lower()
        # Use the same optimizer settings as the server for consistency, if desired
        if optimizer_name == 'sgd':
            return optim.SGD(self.client_model.parameters(), lr=lr)
        elif optimizer_name == 'adam':
            return optim.Adam(self.client_model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def set_model_params(self, global_params: OrderedDict):
        """Updates the local client model (WC) with parameters from the FedServer."""
        self.client_model.load_state_dict(global_params)

    def local_forward_pass(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass on the client model (WC) using one batch of local data.
        Applies Laplacian noise to the activations before returning.

        Returns:
            A tuple containing:
            - noisy_activations (torch.Tensor): The output of WC (Ak,t) with added Laplacian noise.
            - labels (torch.Tensor): The labels for the processed batch.
        """
        try:
            data, labels = next(iter(self.dataloader))
        except StopIteration:
            # Handle case where dataloader is exhausted (e.g., end of epoch)
            # For simplicity in simulation, we might just re-initialize or ignore
            print(f"Client {self.client_id}: Dataloader exhausted. Re-initializing for simulation.")
            # This behavior might need adjustment based on exact training loop design
            self.dataloader = DataLoader(self.dataloader.dataset, batch_size=self.config['batch_size'], shuffle=True)
            data, labels = next(iter(self.dataloader))

        data, labels = data.to(self.device), labels.to(self.device)
        self._data_batch = data # Store for per-sample grad calculation
        self._labels_batch = labels

        # Zero gradients before forward pass
        self.optimizer.zero_grad()

        # Forward pass through client model (WC)
        activations = self.client_model(data)
        # self._activations = activations # Store clean activations (original - incorrect for backward)

        # --- Noise Mechanism 1: Laplacian Noise on Activations --- 
        sensitivity = self.config['dp_noise']['laplacian_sensitivity']
        epsilon_prime = self.config['dp_noise']['epsilon_prime']

        # Add noise, ensuring the noisy tensor requires grad for client backward pass
        if sensitivity > 0:
            noisy_activations = noise_utils.add_laplacian_noise(
                activations,
                sensitivity,
                epsilon_prime,
                device=self.device
            )
        else:
            noisy_activations = activations # No noise added

        # Store the noisy activations (or clean if sensitivity=0) for backward pass
        # Ensure it requires grad and is connected to the original graph.
        self._activations_for_backward = noisy_activations

        # print(f"Client {self.client_id}: Forward pass done. Activation shape: {activations.shape}")
        # Return noisy activations and labels
        # Detach noisy activations before sending to prevent graph issues if server doesn't handle it
        return noisy_activations.detach().clone(), labels.clone()

    def local_backward_pass(self, activation_grads: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Performs the backward pass on the client model (WC) using the gradients
        received from the MainServer.
        Implements per-sample gradient clipping and adds Gaussian noise.

        Args:
            activation_grads (torch.Tensor): The gradient ∇Ak,t received from the MainServer.

        Returns:
            Dict[str, torch.Tensor]: The noisy gradients (∇WCk,t) for the client model parameters.
        """
        # Use the stored activations (potentially noisy) for backward
        if self._activations_for_backward is None or self._data_batch is None:
            raise RuntimeError("Client must perform forward pass before backward pass.")

        # --- Per-Sample Gradient Clipping & Noise (Restored Correct Logic) --- 
        clip_norm = self.config['dp_noise']['clip_norm']
        noise_multiplier = self.config['dp_noise']['noise_multiplier']
        summed_clipped_grads = OrderedDict([(name, torch.zeros_like(param)) for name, param in self.client_model.named_parameters() if param.requires_grad])
        batch_size = self._data_batch.size(0)
        activation_grads = activation_grads.to(self.device)

        # Ensure the shape matches the stored activations
        if activation_grads.shape != self._activations_for_backward.shape:
             print(f"Warning: Client {self.client_id} received activation grads with shape {activation_grads.shape}, expected {self._activations_for_backward.shape}.")
             # Handle potential shape mismatch if necessary

        # Iterate over samples for per-sample gradient clipping
        for i in range(batch_size):
            self.optimizer.zero_grad()
            
            # Use the i-th slice of the stored activations
            # Ensure retain_graph=True ONLY if the graph needs to be preserved after this sample's backward
            # Since we re-do backward for each sample, maybe it's not needed? Check PyTorch docs.
            # For safety and typical per-sample grad implementations, retain_graph=True is often used.
            sample_activation = self._activations_for_backward[i:i+1]
            sample_activation_grad = activation_grads[i:i+1]
            
            # Backward pass for the single sample
            sample_activation.backward(gradient=sample_activation_grad, retain_graph=True)
            
            # Calculate L2 norm of gradients for this sample
            total_norm_sq = torch.zeros(1, device=self.device)
            for name, param in self.client_model.named_parameters():
                if param.grad is not None:
                    # Using param.grad.data might detach, use .grad directly if possible
                    total_norm_sq += param.grad.norm(2).item() ** 2 # Use param.grad directly
                else:
                    # Handle case where a parameter might not get a grad (shouldn't happen here)
                    pass 
            total_norm = torch.sqrt(total_norm_sq)
            
            # Clip gradients
            clip_coef = min(1.0, clip_norm / (total_norm + 1e-6))
            for name, param in self.client_model.named_parameters():
                if param.grad is not None:
                    # Accumulate using .grad directly, not .grad.data to stay within graph if needed? No, summing needs detached values.
                    summed_clipped_grads[name] += param.grad.data * clip_coef # Use .data for accumulation

        # --- Noise Mechanism 2: Gaussian Noise on Aggregated Clipped Gradients --- 
        noisy_gradients = OrderedDict()
        if noise_multiplier > 0:
             for name, summed_grad in summed_clipped_grads.items():
                 noisy_gradients[name] = noise_utils.add_gaussian_noise(
                     summed_grad,
                     clip_norm,
                     noise_multiplier,
                     device=self.device
                 )
             final_gradients_to_send = noisy_gradients
        else:
             # If no noise, send the summed clipped (or unclipped if clip_norm is high) gradients
             final_gradients_to_send = summed_clipped_grads
        # ---------------------------------------------------------------------

        # Clear intermediate values
        self._activations_for_backward = None
        self._data_batch = None 
        self._labels_batch = None 
        self.optimizer.zero_grad() # Zero grads finally before returning

        # Return the NOISY gradients to be sent to FedServer
        return final_gradients_to_send

    # Optional: If clients also update their models locally after backward pass
    # def local_update(self, gradients): # Needs adjustment based on SFL protocol
    #     self.optimizer.zero_grad()
    #     with torch.no_grad():
    #         for name, param in self.client_model.named_parameters():
    #             if name in gradients:
    #                 param.grad = gradients[name]
    #     self.optimizer.step() 