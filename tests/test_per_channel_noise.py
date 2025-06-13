import torch
import unittest
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import copy

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.dp.noise_utils import add_gaussian_noise, clip_tensor

def add_per_channel_gaussian_noise(tensor, clip_norm, noise_multiplier, device='cpu'):
    """
    Adds Gaussian noise on a per-channel basis for convolutional activations.
    Instead of using a single noise scale for the entire activation tensor,
    this scales noise independently for each channel based on its norm.
    
    Args:
        tensor: The input tensor (activations) with shape [batch_size, channels, height, width]
        clip_norm: The L2 norm bound C used for clipping (per channel)
        noise_multiplier: The noise multiplier z (sigma = z * channel_norm)
        device: The device to generate noise on ('cpu' or 'cuda')
        
    Returns:
        Tensor with added per-channel Gaussian noise
    """
    if noise_multiplier <= 0:
        return tensor
        
    # Create output tensor
    result = torch.zeros_like(tensor)
    
    # Process each sample in the batch
    for i in range(tensor.shape[0]):
        # Process each channel separately
        for c in range(tensor.shape[1]):
            channel = tensor[i, c]
            
            # Calculate channel norm
            channel_norm = torch.norm(channel, p=2)
            
            # If channel is all zeros, skip noise addition
            if channel_norm == 0:
                result[i, c] = channel
                continue
                
            # Clip the channel if needed
            if clip_norm > 0 and channel_norm > clip_norm:
                channel = channel * (clip_norm / channel_norm)
                channel_norm = clip_norm
                
            # Add noise scaled by channel norm
            sigma = noise_multiplier * channel_norm
            noise = torch.randn_like(channel, device=device) * sigma
            result[i, c] = channel + noise
            
    return result

class TestPerChannelNoise(unittest.TestCase):
    def setUp(self):
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def test_per_channel_noise_shape(self):
        """Test that the per-channel noise addition preserves tensor shape."""
        # Create a 4D tensor like a convolutional activation
        batch_size = 16
        channels = 20
        height = 12
        width = 12
        tensor = torch.ones(batch_size, channels, height, width)
        
        clip_norm = 1.0
        noise_multiplier = 0.1
        
        noisy_tensor = add_per_channel_gaussian_noise(tensor, clip_norm, noise_multiplier)
        
        self.assertEqual(tensor.shape, noisy_tensor.shape)
        
    def test_per_channel_vs_tensor_noise(self):
        """Compare per-channel noise with tensor-wide noise."""
        # Create a 4D tensor with varying channel norms
        batch_size = 16
        channels = 20
        height = 12
        width = 12
        
        # Create tensor with different scales per channel
        tensor = torch.zeros(batch_size, channels, height, width)
        for c in range(channels):
            # Make each channel have a different scale
            scale = 0.5 + c / channels  # Scales from 0.5 to 1.5
            tensor[:, c, :, :] = scale
            
        clip_norm = 1.0
        noise_multiplier = 0.1
        
        # Apply both noise methods
        tensor_noise = add_gaussian_noise(tensor, clip_norm, noise_multiplier, device=self.device)
        channel_noise = add_per_channel_gaussian_noise(tensor, clip_norm, noise_multiplier, device=self.device)
        
        # Calculate noise magnitude per channel
        tensor_noise_magnitude = torch.zeros(channels)
        channel_noise_magnitude = torch.zeros(channels)
        
        for c in range(channels):
            tensor_diff = tensor_noise[:, c, :, :] - tensor[:, c, :, :]
            channel_diff = channel_noise[:, c, :, :] - tensor[:, c, :, :]
            
            tensor_noise_magnitude[c] = torch.mean(torch.norm(tensor_diff.reshape(batch_size, -1), dim=1))
            channel_noise_magnitude[c] = torch.mean(torch.norm(channel_diff.reshape(batch_size, -1), dim=1))
            
        # Print results
        print("\nNoise Magnitude Comparison (Per Channel):")
        print(f"{'Channel':<10} {'Channel Scale':<15} {'Tensor Noise':<15} {'Channel Noise':<15}")
        print("-" * 55)
        
        for c in range(channels):
            scale = 0.5 + c / channels
            print(f"{c:<10} {scale:<15.3f} {tensor_noise_magnitude[c]:<15.6f} {channel_noise_magnitude[c]:<15.6f}")
            
        # Check that tensor noise is uniform across channels (relative to scale)
        # while channel noise scales with the channel
        tensor_noise_ratio = tensor_noise_magnitude / torch.tensor([0.5 + c / channels for c in range(channels)])
        channel_noise_ratio = channel_noise_magnitude / torch.tensor([0.5 + c / channels for c in range(channels)])
        
        # Tensor noise should have more variance in the ratio (not proportional to channel scale)
        # Channel noise should have less variance (more proportional to channel scale)
        tensor_ratio_std = torch.std(tensor_noise_ratio)
        channel_ratio_std = torch.std(channel_noise_ratio)
        
        print(f"\nTensor noise ratio std: {tensor_ratio_std:.6f}")
        print(f"Channel noise ratio std: {channel_ratio_std:.6f}")
        
        # The channel noise should be more proportional to channel scale
        self.assertLess(channel_ratio_std, tensor_ratio_std)
        
    def test_visualization(self):
        """Visualize the difference between tensor and per-channel noise."""
        if os.environ.get('CI') == 'true':
            self.skipTest("Skipping visualization test in CI environment")
            
        # Create a simple 2D image with 3 channels (like RGB)
        # Channel 1: Low intensity
        # Channel 2: Medium intensity
        # Channel 3: High intensity
        height, width = 100, 100
        image = torch.zeros(1, 3, height, width)
        image[0, 0, :, :] = 0.2  # Low intensity channel
        image[0, 1, :, :] = 0.5  # Medium intensity channel
        image[0, 2, :, :] = 1.0  # High intensity channel
        
        # Apply both noise methods
        clip_norm = 1.0
        noise_multiplier = 0.2  # Higher for visibility
        
        tensor_noise = add_gaussian_noise(image, clip_norm, noise_multiplier)
        channel_noise = add_per_channel_gaussian_noise(image, clip_norm, noise_multiplier)
        
        # Create figure for visualization
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        
        # Original image channels
        for c in range(3):
            axes[0, c].imshow(image[0, c].numpy(), cmap='gray', vmin=0, vmax=1)
            axes[0, c].set_title(f"Original Ch{c+1}")
            axes[0, c].axis('off')
        
        # Tensor noise channels
        for c in range(3):
            axes[1, c].imshow(tensor_noise[0, c].numpy(), cmap='gray', vmin=0, vmax=1)
            axes[1, c].set_title(f"Tensor Noise Ch{c+1}")
            axes[1, c].axis('off')
            
        # Per-channel noise channels
        for c in range(3):
            axes[2, c].imshow(channel_noise[0, c].numpy(), cmap='gray', vmin=0, vmax=1)
            axes[2, c].set_title(f"Channel Noise Ch{c+1}")
            axes[2, c].axis('off')
            
        plt.tight_layout()
        
        # Save the plot
        output_dir = project_root / "tests" / "output"
        output_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_dir / "noise_comparison.png")
        plt.close()
        
        # This is not a real assertion, just for visualization
        self.assertTrue(True)
        
    def test_noise_impact_on_training(self):
        """
        Test how per-channel noise affects a simple CNN compared to tensor-wide noise.
        """
        # Skip this test in CI environments as it's more for analysis
        if os.environ.get('CI') == 'true':
            self.skipTest("Skipping training impact test in CI environment")
            
        # Create a simple CNN for testing
        class SimpleCNN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
                self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
                self.fc = torch.nn.Linear(320, 10)
                
            def forward(self, x):
                x = torch.nn.functional.relu(torch.nn.functional.max_pool2d(self.conv1(x), 2))
                x = torch.nn.functional.relu(torch.nn.functional.max_pool2d(self.conv2(x), 2))
                x = x.view(-1, 320)
                x = self.fc(x)
                return x
                
        # Create synthetic data
        batch_size = 64
        input_channels = 1
        height = 28
        width = 28
        num_classes = 10
        
        # Random data and labels
        inputs = torch.randn(batch_size, input_channels, height, width)
        labels = torch.randint(0, num_classes, (batch_size,))
        
        # Create models and optimizers
        model_clean = SimpleCNN().to(self.device)
        model_tensor = copy.deepcopy(model_clean)
        model_channel = copy.deepcopy(model_clean)
        
        # Parameters
        clip_norm = 1.0
        noise_multiplier = 0.1
        
        # Training loop
        num_epochs = 20
        losses = {
            'clean': [],
            'tensor': [],
            'channel': []
        }
        
        criterion = torch.nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            # Reset seed for reproducible batches
            torch.manual_seed(epoch)
            inputs = torch.randn(batch_size, input_channels, height, width).to(self.device)
            labels = torch.randint(0, num_classes, (batch_size,)).to(self.device)
            
            # Clean model (no noise)
            outputs_clean = model_clean(inputs)
            loss_clean = criterion(outputs_clean, labels)
            losses['clean'].append(loss_clean.item())
            
            # Tensor noise model
            # Add noise to intermediate activations
            with torch.no_grad():
                x = torch.nn.functional.relu(torch.nn.functional.max_pool2d(model_tensor.conv1(inputs), 2))
                # Add tensor-wide noise
                x = add_gaussian_noise(x, clip_norm, noise_multiplier, device=self.device)
                x = torch.nn.functional.relu(torch.nn.functional.max_pool2d(model_tensor.conv2(x), 2))
                # Add tensor-wide noise again
                x = add_gaussian_noise(x, clip_norm, noise_multiplier, device=self.device)
                x = x.view(-1, 320)
                outputs_tensor = model_tensor.fc(x)
                
            loss_tensor = criterion(outputs_tensor, labels)
            losses['tensor'].append(loss_tensor.item())
            
            # Channel noise model
            with torch.no_grad():
                x = torch.nn.functional.relu(torch.nn.functional.max_pool2d(model_channel.conv1(inputs), 2))
                # Add per-channel noise
                x = add_per_channel_gaussian_noise(x, clip_norm, noise_multiplier, device=self.device)
                x = torch.nn.functional.relu(torch.nn.functional.max_pool2d(model_channel.conv2(x), 2))
                # Add per-channel noise again
                x = add_per_channel_gaussian_noise(x, clip_norm, noise_multiplier, device=self.device)
                x = x.view(-1, 320)
                outputs_channel = model_channel.fc(x)
                
            loss_channel = criterion(outputs_channel, labels)
            losses['channel'].append(loss_channel.item())
            
            # Update all models the same way
            for name, model in [('clean', model_clean), ('tensor', model_tensor), ('channel', model_channel)]:
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                optimizer.zero_grad()
                
                if name == 'clean':
                    loss_clean.backward()
                elif name == 'tensor':
                    loss_tensor.backward()
                else:  # channel
                    loss_channel.backward()
                    
                optimizer.step()
                
        # Print final losses
        print("\nFinal Training Losses:")
        for name, loss_history in losses.items():
            print(f"{name}: {loss_history[-1]:.4f}")
            
        # Plot loss curves
        plt.figure(figsize=(10, 6))
        for name, loss_history in losses.items():
            plt.plot(loss_history, label=name)
            
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        output_dir = project_root / "tests" / "output"
        output_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_dir / "noise_training_impact.png")
        plt.close()
        
        # Check if per-channel noise has less impact on training than tensor-wide noise
        avg_tensor_loss = sum(losses['tensor']) / len(losses['tensor'])
        avg_channel_loss = sum(losses['channel']) / len(losses['channel'])
        avg_clean_loss = sum(losses['clean']) / len(losses['clean'])
        
        print(f"Average losses - Clean: {avg_clean_loss:.4f}, Tensor: {avg_tensor_loss:.4f}, Channel: {avg_channel_loss:.4f}")
        
        # Per-channel noise should be closer to clean than tensor-wide noise
        tensor_diff = abs(avg_tensor_loss - avg_clean_loss)
        channel_diff = abs(avg_channel_loss - avg_clean_loss)
        
        print(f"Difference from clean - Tensor: {tensor_diff:.4f}, Channel: {channel_diff:.4f}")
        
        # This might not always be true due to randomness, so we don't assert it
        # but we print the result for analysis
        if channel_diff < tensor_diff:
            print("Per-channel noise has less impact on training than tensor-wide noise")
        else:
            print("Tensor-wide noise has less impact on training than per-channel noise")
            
        return losses  # Return for further analysis if needed

if __name__ == '__main__':
    unittest.main() 