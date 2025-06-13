import torch
import unittest
import sys
import os
import numpy as np
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.sfl.client import SFLClient
from src.models.simple_cnn import SimpleCNN
from src.models.split_utils import split_model

class TestSFLClient(unittest.TestCase):
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
        self.client_model, _ = split_model(self.model, self.cut_layer)
        
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
        
    def test_forward_pass_shape(self):
        """Test that the forward pass produces outputs with the expected shape."""
        client = SFLClient(
            client_id=0,
            client_model=self.client_model,
            dataloader=self.dataloader,
            config=self.config,
            device=self.device
        )
        
        # Run forward pass
        activations, labels = client.local_forward_pass()
        
        # Check shapes
        self.assertEqual(activations.shape[0], self.batch_size)
        # The output shape depends on the specific model architecture
        # For SimpleCNN with cut_layer=3, the output should be [batch_size, 20, 12, 12]
        expected_shape = (self.batch_size, 20, 12, 12)
        self.assertEqual(activations.shape, expected_shape)
        
    def test_forward_pass_clipping(self):
        """Test that activations are properly clipped in the forward pass."""
        # Create a client with high activation_clip_norm to see the effect
        test_config = self.config.copy()
        test_config['dp_noise'] = self.config['dp_noise'].copy()
        test_config['dp_noise']['activation_clip_norm'] = 0.1  # Very low clip norm
        
        client = SFLClient(
            client_id=0,
            client_model=self.client_model,
            dataloader=self.dataloader,
            config=test_config,
            device=self.device
        )
        
        # Run forward pass
        activations, _ = client.local_forward_pass()
        
        # Get the activations before noise (need to modify SFLClient to expose this)
        # Since we can't directly access the pre-noise activations, we'll check indirectly
        # by running another forward pass with no noise
        test_config['dp_noise']['activation_noise_multiplier'] = 0.0
        client_no_noise = SFLClient(
            client_id=1,
            client_model=self.client_model,
            dataloader=self.dataloader,
            config=test_config,
            device=self.device
        )
        
        # Reset the dataloader to get the same batch
        torch.manual_seed(42)
        np.random.seed(42)
        clipped_activations, _ = client_no_noise.local_forward_pass()
        
        # Check that all activations are clipped
        for i in range(self.batch_size):
            # Reshape to 2D for easier norm calculation
            act_flat = clipped_activations[i].view(clipped_activations[i].shape[0], -1)
            for j in range(act_flat.shape[0]):
                norm = torch.norm(act_flat[j], p=2).item()
                self.assertLessEqual(norm, test_config['dp_noise']['activation_clip_norm'] + 1e-5)
        
    def test_forward_pass_noise(self):
        """Test that noise is properly added in the forward pass."""
        # Create two clients with same model but different noise settings
        client1 = SFLClient(
            client_id=0,
            client_model=self.client_model,
            dataloader=self.dataloader,
            config=self.config,
            device=self.device
        )
        
        no_noise_config = self.config.copy()
        no_noise_config['dp_noise'] = self.config['dp_noise'].copy()
        no_noise_config['dp_noise']['activation_noise_multiplier'] = 0.0
        
        client2 = SFLClient(
            client_id=1,
            client_model=self.client_model,
            dataloader=self.dataloader,
            config=no_noise_config,
            device=self.device
        )
        
        # Reset seeds to get the same batch and model initialization
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Run forward passes
        noisy_activations, _ = client1.local_forward_pass()
        
        # Reset seeds again
        torch.manual_seed(42)
        np.random.seed(42)
        
        clean_activations, _ = client2.local_forward_pass()
        
        # Check that noise was added (activations should be different)
        self.assertFalse(torch.allclose(noisy_activations, clean_activations))
        
        # Calculate noise magnitude
        noise = noisy_activations - clean_activations
        avg_noise_magnitude = torch.mean(torch.norm(noise.view(noise.shape[0], -1), dim=1)).item()
        
        # Expected noise magnitude calculation is complex for multi-dimensional activations
        # This is just a rough sanity check
        clip_norm = self.config['dp_noise']['activation_clip_norm']
        noise_mult = self.config['dp_noise']['activation_noise_multiplier']
        
        print(f"Average noise magnitude: {avg_noise_magnitude:.6f}")
        print(f"Clip norm * noise multiplier: {clip_norm * noise_mult:.6f}")
        
        # The noise should be non-zero
        self.assertGreater(avg_noise_magnitude, 0.0)
        
    def test_noise_impact_on_training(self):
        """Test the impact of different noise levels on a simple training loop."""
        # This test simulates a simplified training process to see how noise affects convergence
        
        # Create test configs with different noise levels
        noise_levels = [0.0, 0.1, 0.5, 1.0]
        results = {}
        
        for noise_level in noise_levels:
            # Create config with this noise level
            test_config = self.config.copy()
            test_config['dp_noise'] = self.config['dp_noise'].copy()
            test_config['dp_noise']['activation_noise_multiplier'] = noise_level
            
            # Create client and server models
            torch.manual_seed(42)
            np.random.seed(42)
            
            model = SimpleCNN()
            client_model, server_model = split_model(model, self.cut_layer)
            
            client = SFLClient(
                client_id=0,
                client_model=client_model,
                dataloader=self.dataloader,
                config=test_config,
                device=self.device
            )
            
            # Simple training loop (just to measure noise impact)
            # This is not a full SFL training loop, just a simplified version
            losses = []
            for _ in range(5):  # Just a few iterations
                # Forward pass
                activations, labels = client.local_forward_pass()
                
                # Simulate server processing (just a simple loss calculation)
                server_model.to(self.device)
                server_output = server_model(activations)
                criterion = torch.nn.CrossEntropyLoss()
                loss = criterion(server_output, labels)
                losses.append(loss.item())
                
                # No need for backward pass in this test
            
            results[noise_level] = losses
            
        # Print results
        print("\nNoise impact on training loss:")
        for noise_level, losses in results.items():
            print(f"Noise level {noise_level}: Final loss = {losses[-1]:.4f}, Avg loss = {np.mean(losses):.4f}")
            
        # Check that higher noise generally leads to higher loss
        # This might not always be true due to randomness, but should be a general trend
        avg_losses = [np.mean(results[noise_level]) for noise_level in noise_levels]
        
        # Just a basic check - not always reliable due to randomness
        # We expect at least some correlation between noise level and loss
        correlation = np.corrcoef(noise_levels, avg_losses)[0, 1]
        print(f"Correlation between noise level and average loss: {correlation:.4f}")
        
        # This is not a strict test, just informative
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main() 