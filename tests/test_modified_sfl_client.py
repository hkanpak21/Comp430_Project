import torch
import unittest
import sys
import os
import numpy as np
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
import copy

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.sfl.client import SFLClient
from src.models.simple_cnn import SimpleCNN
from src.models.split_utils import split_model
from src.dp import noise_utils

class ModifiedSFLClient(SFLClient):
    """
    A modified version of SFLClient for testing purposes.
    This version allows us to:
    1. Expose intermediate activations before noise
    2. Apply different noise strategies
    3. Control the noise magnitude more precisely
    """
    
    def __init__(self, client_id, client_model, dataloader, config, device, 
                 noise_strategy='gaussian', noise_scale_factor=1.0):
        """
        Args:
            noise_strategy: One of ['gaussian', 'none', 'reduced', 'per_channel']
            noise_scale_factor: Factor to multiply the noise by (for 'reduced')
        """
        super().__init__(client_id, client_model, dataloader, config, device)
        self.noise_strategy = noise_strategy
        self.noise_scale_factor = noise_scale_factor
        self.raw_activations = None  # Store for testing
        
    def local_forward_pass(self):
        """
        Modified forward pass with different noise strategies.
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
        self.raw_activations = activations.clone()  # Store raw activations for testing

        # Apply clipping
        if self._activation_clip_norm > 0:
            clipped_activations = torch.zeros_like(activations)
            for i in range(activations.shape[0]):
                clipped_activations[i] = noise_utils.clip_tensor(
                    activations[i], 
                    self._activation_clip_norm
                )
            activations = clipped_activations
            
        # Apply noise based on strategy
        if self.noise_strategy == 'none':
            # No noise
            noisy_activations = activations
            
        elif self.noise_strategy == 'reduced':
            # Reduced noise (scaled down)
            if self._activation_noise_multiplier > 0:
                reduced_multiplier = self._activation_noise_multiplier * self.noise_scale_factor
                noisy_activations = noise_utils.add_gaussian_noise(
                    activations,
                    self._activation_clip_norm,
                    reduced_multiplier,
                    device=self.device
                )
            else:
                noisy_activations = activations
                
        elif self.noise_strategy == 'per_channel':
            # Apply noise per channel instead of per activation tensor
            # This might be more appropriate for convolutional layers
            noisy_activations = torch.zeros_like(activations)
            
            # For each sample in the batch
            for i in range(activations.shape[0]):
                # For each channel
                for c in range(activations.shape[1]):
                    channel = activations[i, c]
                    channel_norm = torch.norm(channel, p=2)
                    
                    # Scale the noise to the channel norm
                    if channel_norm > 0 and self._activation_noise_multiplier > 0:
                        # Generate noise with std = noise_multiplier * channel_norm
                        noise = torch.randn_like(channel) * self._activation_noise_multiplier * channel_norm
                        noisy_activations[i, c] = channel + noise
                    else:
                        noisy_activations[i, c] = channel
                        
        else:  # Default: 'gaussian'
            # Standard Gaussian noise as in the original implementation
            if self._activation_noise_multiplier > 0:
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


class TestModifiedSFLClient(unittest.TestCase):
    def setUp(self):
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create a simple dataset for testing
        self.batch_size = 16
        self.input_shape = (1, 28, 28)  # MNIST-like
        self.num_samples = 32
        
        # Create random data
        data = torch.randn(self.num_samples, *self.input_shape)
        labels = torch.randint(0, 10, (self.num_samples,))
        dataset = TensorDataset(data, labels)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Create a model and split it
        self.model = SimpleCNN()
        self.cut_layer = 3  # Split after the first conv layer
        self.client_model, self.server_model = split_model(self.model, self.cut_layer)
        
        # Basic config
        self.config = {
            'batch_size': self.batch_size,
            'lr': 0.001,
            'optimizer': 'Adam',
            'dp_noise': {
                'activation_clip_norm': 1.0,
                'activation_noise_multiplier': 0.1,
                'clip_norm': 1.0,
                'initial_sigma': 0.1,
                'adaptive_clipping_factor': 0.0
            }
        }
    
    def test_noise_strategies(self):
        """Compare the impact of different noise strategies on activations."""
        strategies = {
            'none': 0.0,      # No noise
            'gaussian': 1.0,  # Standard Gaussian noise
            'reduced': 0.5,   # Reduced Gaussian noise
            'per_channel': 1.0  # Per-channel noise
        }
        
        # Store results for comparison
        results = {}
        
        # Create a reference client with no noise for comparison
        ref_client = ModifiedSFLClient(
            client_id=99,
            client_model=copy.deepcopy(self.client_model),
            dataloader=self.dataloader,
            config=self.config,
            device=self.device,
            noise_strategy='none'
        )
        
        # Reset seed for consistent data
        torch.manual_seed(42)
        np.random.seed(42)
        ref_activations, ref_labels = ref_client.local_forward_pass()
        
        # Test each strategy
        for strategy, scale in strategies.items():
            # Create client with this strategy
            client = ModifiedSFLClient(
                client_id=0,
                client_model=copy.deepcopy(self.client_model),
                dataloader=self.dataloader,
                config=self.config,
                device=self.device,
                noise_strategy=strategy,
                noise_scale_factor=scale
            )
            
            # Reset seed for consistent data
            torch.manual_seed(42)
            np.random.seed(42)
            
            # Run forward pass
            activations, labels = client.local_forward_pass()
            
            # Calculate noise magnitude
            if strategy != 'none':
                noise = activations - ref_activations
                avg_noise_magnitude = torch.mean(torch.norm(noise.view(noise.shape[0], -1), dim=1)).item()
            else:
                avg_noise_magnitude = 0.0
                
            # Store results
            results[strategy] = {
                'activations': activations,
                'noise_magnitude': avg_noise_magnitude
            }
            
            # Run server forward pass to get loss
            self.server_model.to(self.device)
            server_output = self.server_model(activations)
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(server_output, labels)
            results[strategy]['loss'] = loss.item()
            
        # Print results
        print("\nNoise Strategy Comparison:")
        print(f"{'Strategy':<12} {'Noise Magnitude':<20} {'Loss':<10}")
        print("-" * 42)
        for strategy, data in results.items():
            print(f"{strategy:<12} {data['noise_magnitude']:<20.6f} {data['loss']:<10.4f}")
            
        # Check that 'none' strategy has zero noise
        self.assertAlmostEqual(results['none']['noise_magnitude'], 0.0)
        
        # Check that 'reduced' has less noise than 'gaussian'
        self.assertLess(results['reduced']['noise_magnitude'], results['gaussian']['noise_magnitude'])
        
        # Check if per_channel noise is different from gaussian
        self.assertNotEqual(results['per_channel']['noise_magnitude'], results['gaussian']['noise_magnitude'])
        
    def test_training_with_different_strategies(self):
        """Test a simple training loop with different noise strategies."""
        strategies = {
            'none': 0.0,      # No noise
            'gaussian': 1.0,  # Standard Gaussian noise
            'reduced': 0.2,   # Significantly reduced noise
            'per_channel': 1.0  # Per-channel noise
        }
        
        # Store results
        training_results = {}
        
        for strategy, scale in strategies.items():
            # Create client and server models
            torch.manual_seed(42)
            np.random.seed(42)
            
            client_model = copy.deepcopy(self.client_model)
            server_model = copy.deepcopy(self.server_model).to(self.device)
            
            client = ModifiedSFLClient(
                client_id=0,
                client_model=client_model,
                dataloader=self.dataloader,
                config=self.config,
                device=self.device,
                noise_strategy=strategy,
                noise_scale_factor=scale
            )
            
            # Simple training loop
            losses = []
            for epoch in range(10):  # More epochs for better trend
                # Forward pass
                activations, labels = client.local_forward_pass()
                
                # Server forward and backward
                server_model.zero_grad()
                server_output = server_model(activations)
                criterion = torch.nn.CrossEntropyLoss()
                loss = criterion(server_output, labels)
                loss.backward()
                
                # Server optimizer step
                with torch.no_grad():
                    for param in server_model.parameters():
                        if param.grad is not None:
                            param.data -= 0.01 * param.grad  # Simple SGD
                
                losses.append(loss.item())
                
            training_results[strategy] = losses
            
        # Plot training curves
        print("\nTraining Loss by Noise Strategy:")
        for strategy, losses in training_results.items():
            print(f"{strategy}: Initial loss = {losses[0]:.4f}, Final loss = {losses[-1]:.4f}")
            
        # Check if no noise strategy converges better
        self.assertLess(training_results['none'][-1], training_results['gaussian'][-1])
        
        # Check if reduced noise is better than full noise
        self.assertLess(training_results['reduced'][-1], training_results['gaussian'][-1])
        
        return training_results  # Return for potential plotting

if __name__ == '__main__':
    unittest.main() 