import torch

def add_gaussian_noise(tensor, clip_norm, noise_multiplier, device='cpu'):
    """
    Adds Gaussian noise scaled by clip_norm and noise_multiplier.
    Used for both activations and gradients in the unified DP approach.

    Args:
        tensor: The input tensor (e.g., clipped activations or summed clipped gradients).
        clip_norm: The L2 norm bound C used for clipping.
        noise_multiplier: The noise multiplier z (sigma = z * C).
        device: The device to generate noise on ('cpu' or 'cuda').

    Returns:
        Tensor with added Gaussian noise.
    """
    if noise_multiplier < 0:
        raise ValueError("Noise multiplier cannot be negative.")
    if clip_norm <= 0:
        # Allow clip_norm=0 for cases where no privacy is applied (noise_multiplier=0)
        if noise_multiplier > 0:
             raise ValueError("Clip norm must be positive if noise multiplier is positive.")
        else:
            # No clipping, no noise - return original tensor
            return tensor

    sigma = noise_multiplier * clip_norm

    # If sigma is zero, no noise is added.
    if sigma == 0.0:
        return tensor

    # Generate Gaussian noise N(0, sigma^2 * I)
    gaussian_dist = torch.distributions.normal.Normal(loc=0.0, scale=sigma)
    noise = gaussian_dist.sample(tensor.size()).to(device)

    return tensor + noise

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

def add_reduced_gaussian_noise(tensor, clip_norm, noise_multiplier, reduction_factor=0.5, device='cpu'):
    """
    Adds Gaussian noise with a reduced magnitude compared to standard DP.
    This is a compromise approach that maintains some privacy while reducing
    the impact on model performance.
    
    Args:
        tensor: The input tensor (e.g., clipped activations or summed clipped gradients).
        clip_norm: The L2 norm bound C used for clipping.
        noise_multiplier: The noise multiplier z (sigma = z * C).
        reduction_factor: Factor to reduce the noise by (default: 0.5).
        device: The device to generate noise on ('cpu' or 'cuda').
        
    Returns:
        Tensor with added reduced Gaussian noise.
    """
    # Simply scale down the noise multiplier
    return add_gaussian_noise(tensor, clip_norm, noise_multiplier * reduction_factor, device)

def clip_gradients(model, clip_norm):
    """
    Clips the L2 norm of gradients for each parameter in the model.
    Note: This function clips the *aggregated* gradients attached to model.parameters().
          For per-sample clipping as required by Mechanism 2, a different approach
          (micro-batching loop during backward pass) is needed within the client logic.
          This function is provided as a standard gradient clipping utility, but is
          NOT the one used for the per-sample clipping requirement.
    """
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
    return total_norm

def clip_tensor(tensor, clip_norm):
    """
    Clips a tensor by its L2 norm.
    
    Args:
        tensor: The tensor to clip
        clip_norm: Maximum L2 norm
        
    Returns:
        Clipped tensor
    """
    norm = torch.norm(tensor, p=2)
    if norm > clip_norm:
        tensor = tensor * (clip_norm / norm)
    return tensor

def clip_per_channel(tensor, clip_norm):
    """
    Clips each channel of a tensor independently by its L2 norm.
    
    Args:
        tensor: The tensor to clip with shape [batch_size, channels, height, width]
        clip_norm: Maximum L2 norm per channel
        
    Returns:
        Tensor with each channel clipped
    """
    result = torch.zeros_like(tensor)
    
    # Process each sample in the batch
    for i in range(tensor.shape[0]):
        # Process each channel separately
        for c in range(tensor.shape[1]):
            channel = tensor[i, c]
            channel_norm = torch.norm(channel, p=2)
            
            if channel_norm > clip_norm:
                result[i, c] = channel * (clip_norm / channel_norm)
            else:
                result[i, c] = channel
                
    return result 