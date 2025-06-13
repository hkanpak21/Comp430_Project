import torch
import unittest
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.dp.noise_utils import add_gaussian_noise, clip_tensor

class TestNoiseUtils(unittest.TestCase):
    def setUp(self):
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def test_add_gaussian_noise_shape(self):
        """Test that the noise addition preserves tensor shape."""
        tensor = torch.ones(10, 20)
        clip_norm = 1.0
        noise_multiplier = 0.1
        
        noisy_tensor = add_gaussian_noise(tensor, clip_norm, noise_multiplier)
        
        self.assertEqual(tensor.shape, noisy_tensor.shape)
        
    def test_add_gaussian_noise_zero_multiplier(self):
        """Test that zero noise multiplier doesn't change the tensor."""
        tensor = torch.ones(10, 20)
        clip_norm = 1.0
        noise_multiplier = 0.0
        
        noisy_tensor = add_gaussian_noise(tensor, clip_norm, noise_multiplier)
        
        torch.testing.assert_close(tensor, noisy_tensor)
        
    def test_add_gaussian_noise_statistics(self):
        """Test that the added noise follows expected Gaussian statistics."""
        # Create a large tensor to get good statistics
        tensor = torch.zeros(10000)
        clip_norm = 1.0
        noise_multiplier = 0.1
        expected_std = clip_norm * noise_multiplier
        
        noisy_tensor = add_gaussian_noise(tensor, clip_norm, noise_multiplier)
        noise = noisy_tensor - tensor
        
        # Check mean close to 0
        self.assertAlmostEqual(noise.mean().item(), 0.0, delta=0.01)
        
        # Check std close to expected
        self.assertAlmostEqual(noise.std().item(), expected_std, delta=0.01)
        
    def test_clip_tensor(self):
        """Test that tensor clipping works correctly."""
        # Create a tensor with norm > clip_norm
        tensor = torch.tensor([3.0, 4.0])  # norm = 5
        clip_norm = 1.0
        
        clipped_tensor = clip_tensor(tensor, clip_norm)
        
        # Check norm is now equal to clip_norm
        self.assertAlmostEqual(torch.norm(clipped_tensor, p=2).item(), clip_norm, delta=1e-6)
        
        # Check direction is preserved
        normalized_original = tensor / torch.norm(tensor, p=2)
        normalized_clipped = clipped_tensor / torch.norm(clipped_tensor, p=2)
        torch.testing.assert_close(normalized_original, normalized_clipped)
        
    def test_noise_distribution_visualization(self):
        """Visualize the noise distribution (not an actual test)."""
        # Skip this test in CI environments
        if os.environ.get('CI') == 'true':
            self.skipTest("Skipping visualization test in CI environment")
            
        # Create a large tensor for visualization
        tensor = torch.zeros(100000)
        clip_norm = 1.0
        noise_multiplier = 0.1
        
        noisy_tensor = add_gaussian_noise(tensor, clip_norm, noise_multiplier)
        noise = noisy_tensor.numpy()
        
        # Create histogram
        plt.figure(figsize=(10, 6))
        plt.hist(noise, bins=100, density=True, alpha=0.7)
        
        # Plot the expected Gaussian PDF
        x = np.linspace(-0.5, 0.5, 1000)
        expected_std = clip_norm * noise_multiplier
        pdf = 1/(expected_std * np.sqrt(2*np.pi)) * np.exp(-0.5*((x)/expected_std)**2)
        plt.plot(x, pdf, 'r-', linewidth=2)
        
        plt.title(f'Noise Distribution (σ={expected_std:.3f})')
        plt.xlabel('Noise Value')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        output_dir = project_root / "tests" / "output"
        output_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_dir / "noise_distribution.png")
        plt.close()
        
        # This is not a real assertion, just for visualization
        self.assertTrue(True)
        
    def test_activation_noise_simulation(self):
        """Simulate the activation noise process in SFLClient to validate behavior."""
        # Create a batch of "activations"
        batch_size = 64
        feature_dim = 20
        activations = torch.randn(batch_size, feature_dim, device=self.device)
        
        # Parameters from config
        activation_clip_norm = 1.0
        activation_noise_multiplier = 0.1
        
        # Simulate the SFLClient.local_forward_pass clipping and noise addition
        clipped_activations = torch.zeros_like(activations)
        for i in range(activations.shape[0]):
            clipped_activations[i] = clip_tensor(
                activations[i], 
                activation_clip_norm
            )
            
        # Apply Gaussian noise to clipped activations
        noisy_activations = add_gaussian_noise(
            clipped_activations,
            activation_clip_norm,
            activation_noise_multiplier,
            device=self.device
        )
        
        # Check that all samples are clipped
        for i in range(batch_size):
            self.assertLessEqual(
                torch.norm(clipped_activations[i], p=2).item(),
                activation_clip_norm + 1e-5  # Add small epsilon for numerical stability
            )
        
        # Check that noise has been added
        self.assertFalse(torch.allclose(clipped_activations, noisy_activations))
        
        # Calculate average noise magnitude
        noise = noisy_activations - clipped_activations
        avg_noise_magnitude = torch.mean(torch.norm(noise, dim=1)).item()
        expected_noise_magnitude = activation_noise_multiplier * activation_clip_norm * np.sqrt(feature_dim)
        
        # The average noise magnitude should be close to the expected value
        # (allowing for some statistical variation)
        print(f"Average noise magnitude: {avg_noise_magnitude:.4f}")
        print(f"Expected noise magnitude: {expected_noise_magnitude:.4f}")
        
        # This is more of a sanity check than a strict test
        # We expect the noise magnitude to be in the same ballpark as the expected value
        self.assertLess(abs(avg_noise_magnitude - expected_noise_magnitude), 
                        expected_noise_magnitude)  # Within 100% of expected

if __name__ == '__main__':
    unittest.main() 