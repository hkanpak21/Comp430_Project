import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import OrderedDict
from typing import Tuple, Dict

# Import the improved noise utilities
from ..dp import improved_noise_utils as noise_utils

class ImprovedSFLClient:
    """
    Improved version of SFLClient with better noise handling options.
    Manages local data, client-side model (WC), performs local computations,
    applies noise, and communicates intermediate results.
    """
    def __init__(self, client_id: int, client_model: nn.Module, dataloader: DataLoader, 
                 config: dict, device: torch.device, noise_mode: str = 'tensor'):
        """
        Args:
            client_id: Unique identifier for the client.
            client_model: A *copy* of the initial client-side model (WC) architecture.
            dataloader: DataLoader for the client's local dataset partition.
            config: Configuration dictionary.
            device: The torch device ('cpu' or 'cuda').
            noise_mode: Type of noise to apply ('tensor', 'channel', 'reduced', 'none').
        """
        self.client_id = client_id
        self.client_model = client_model.to(device)
        self.dataloader = dataloader
        self.config = config
        self.device = device
        self.optimizer = self._create_optimizer() # Optimizer for WC
        self.noise_mode = noise_mode

        # Store intermediate activation for backward pass
        self._activations = None
        self._data_batch = None # Store data batch to access individual samples for clipping
        self._labels_batch = None # Store labels corresponding to the data batch

        # Adaptive DP: Store previous round's gradient norms and current noise scale
        self._prev_round_grad_norms = None
        self._current_clip_threshold = config['dp_noise']['clip_norm'] # Initial threshold
        self._current_sigma = config['dp_noise']['initial_sigma'] # Initial noise scale
        
        # Activation clipping and noise parameters
        self._activation_clip_norm = config['dp_noise'].get('activation_clip_norm', 1.0)
        self._activation_noise_multiplier = config['dp_noise'].get('activation_noise_multiplier', 1.0)
        
        # Optional noise reduction factor (used with 'reduced' mode)
        self._noise_reduction_factor = config['dp_noise'].get('noise_reduction_factor', 0.5)

    def _create_optimizer(self) -> optim.Optimizer:
        """Creates the optimizer for the client-side model (WC)."""
        lr = self.config.get('lr', 0.01)
        optimizer_name = self.config.get('optimizer', 'SGD').lower()
        if optimizer_name == 'sgd':
            return optim.SGD(self.client_model.parameters(), lr=lr)
        elif optimizer_name == 'adam':
            return optim.Adam(self.client_model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def set_model_params(self, global_params: OrderedDict):
        """Updates the local client model (WC) with parameters from the FedServer."""
        self.client_model.load_state_dict(global_params)

    def update_noise_scale(self, new_sigma: float):
        """Updates the current noise scale sigma_t."""
        self._current_sigma = new_sigma

    def set_noise_mode(self, mode: str):
        """
        Change the noise mode at runtime.
        
        Args:
            mode: One of 'tensor', 'channel', 'reduced', 'none'
        """
        valid_modes = ['tensor', 'channel', 'reduced', 'none']
        if mode not in valid_modes:
            raise ValueError(f"Invalid noise mode '{mode}'. Must be one of {valid_modes}")
        self.noise_mode = mode

    def _calculate_adaptive_clip_threshold(self) -> float:
        """Calculates the adaptive clipping threshold Ck_t for the current round."""
        # Check if adaptive clipping is disabled (adaptive_clipping_factor = 0.0)
        if self.config['dp_noise']['adaptive_clipping_factor'] == 0.0:
            # For fixed DP, always use the initial clip norm
            return self.config['dp_noise']['clip_norm']
            
        if self._prev_round_grad_norms is None:
            # First round: use initial threshold
            return self.config['dp_noise']['clip_norm']
        
        # Calculate mean norm from previous round
        mean_norm = torch.mean(torch.tensor(self._prev_round_grad_norms, device=self.device))
        # Apply adaptive factor
        adaptive_factor = self.config['dp_noise']['adaptive_clipping_factor']
        return float(adaptive_factor * mean_norm)

    def local_forward_pass(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass on the client model (WC) using one batch of local data.
        Clips activations and applies noise according to the selected noise_mode.
        """
        try:
            data, labels = next(iter(self.dataloader))
        except StopIteration:
            print(f"Client {self.client_id}: Dataloader exhausted. Re-initializing for simulation.")
            self.dataloader = DataLoader(self.dataloader.dataset, batch_size=self.config['batch_size'], shuffle=True)
            data, labels = next(iter(self.dataloader))

        data, labels = data.to(self.device), labels.to(self.device)
        self._data_batch = data
        self._labels_batch = labels

        self.optimizer.zero_grad()
        activations = self.client_model(data)

        # Apply clipping based on noise mode
        if self._activation_clip_norm > 0:
            if self.noise_mode == 'channel':
                # Clip each channel independently
                activations = noise_utils.clip_per_channel(
                    activations, 
                    self._activation_clip_norm
                )
            else:
                # Standard tensor-wise clipping
                clipped_activations = torch.zeros_like(activations)
                for i in range(activations.shape[0]):
                    clipped_activations[i] = noise_utils.clip_tensor(
                        activations[i], 
                        self._activation_clip_norm
                    )
                activations = clipped_activations

        # Apply noise based on selected mode
        if self._activation_noise_multiplier > 0:
            if self.noise_mode == 'none':
                # No noise
                noisy_activations = activations
            elif self.noise_mode == 'channel':
                # Per-channel noise
                noisy_activations = noise_utils.add_per_channel_gaussian_noise(
                    activations,
                    self._activation_clip_norm,
                    self._activation_noise_multiplier,
                    device=self.device
                )
            elif self.noise_mode == 'reduced':
                # Reduced noise
                noisy_activations = noise_utils.add_reduced_gaussian_noise(
                    activations,
                    self._activation_clip_norm,
                    self._activation_noise_multiplier,
                    reduction_factor=self._noise_reduction_factor,
                    device=self.device
                )
            else:  # 'tensor' (default)
                # Standard tensor-wide Gaussian noise
                noisy_activations = noise_utils.add_gaussian_noise(
                    activations,
                    self._activation_clip_norm,
                    self._activation_noise_multiplier,
                    device=self.device
                )
        else:
            noisy_activations = activations

        self._activations_for_backward = noisy_activations
        return noisy_activations.detach().clone(), labels.clone()

    def local_backward_pass(self, activation_grads: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Performs the backward pass with adaptive clipping and noise.
        """
        if self._activations_for_backward is None or self._data_batch is None:
            raise RuntimeError("Client must perform forward pass before backward pass.")

        # Calculate adaptive clipping threshold for this round
        self._current_clip_threshold = self._calculate_adaptive_clip_threshold()
        
        summed_clipped_grads = OrderedDict([(name, torch.zeros_like(param)) 
                                          for name, param in self.client_model.named_parameters() 
                                          if param.requires_grad])
        
        batch_size = self._data_batch.size(0)
        activation_grads = activation_grads.to(self.device)
        current_round_grad_norms = [] # Store norms for next round's threshold calculation

        # Per-sample gradient computation with adaptive clipping
        for i in range(batch_size):
            self.optimizer.zero_grad()
            sample_activation = self._activations_for_backward[i:i+1]
            sample_activation_grad = activation_grads[i:i+1]
            
            sample_activation.backward(gradient=sample_activation_grad, retain_graph=True)
            
            # Calculate L2 norm of gradients for this sample
            total_norm_sq = torch.zeros(1, device=self.device)
            for name, param in self.client_model.named_parameters():
                if param.grad is not None:
                    total_norm_sq += param.grad.norm(2).item() ** 2
            
            total_norm = torch.sqrt(total_norm_sq)
            current_round_grad_norms.append(total_norm.item())
            
            # Clip gradients using adaptive threshold
            clip_coef = min(1.0, self._current_clip_threshold / (total_norm + 1e-6))
            
            for name, param in self.client_model.named_parameters():
                if param.grad is not None:
                    summed_clipped_grads[name] += param.grad.data * clip_coef

        # Store gradient norms for next round's threshold calculation
        self._prev_round_grad_norms = current_round_grad_norms

        # Add noise based on selected mode
        noisy_gradients = OrderedDict()
        if self._current_sigma > 0:
            if self.noise_mode == 'none':
                # No noise
                noisy_gradients = summed_clipped_grads
            elif self.noise_mode == 'reduced':
                # Reduced noise
                for name, summed_grad in summed_clipped_grads.items():
                    noisy_gradients[name] = noise_utils.add_reduced_gaussian_noise(
                        summed_grad,
                        self._current_clip_threshold,
                        self._current_sigma,
                        reduction_factor=self._noise_reduction_factor,
                        device=self.device
                    )
            else:  # 'tensor' or 'channel' (both use the same gradient noise)
                # Standard tensor-wide Gaussian noise for gradients
                for name, summed_grad in summed_clipped_grads.items():
                    noisy_gradients[name] = noise_utils.add_gaussian_noise(
                        summed_grad,
                        self._current_clip_threshold,
                        self._current_sigma,
                        device=self.device
                    )
        else:
            noisy_gradients = summed_clipped_grads

        # Clear intermediate values
        self._activations_for_backward = None
        self._data_batch = None
        self._labels_batch = None
        self.optimizer.zero_grad()

        return noisy_gradients 