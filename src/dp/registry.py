from .privacy_accountant import ManualPrivacyAccountant

class VanillaGaussianAccountant(ManualPrivacyAccountant):
    """
    Same RDP machinery but σ is *frozen* at construction.
    """
    def __init__(self, noise_multiplier, sampling_rate,
                 moment_orders=None):
        super().__init__(moment_orders)
        self.fixed_sigma = noise_multiplier
        self.q = sampling_rate
        self._eps_history = []  # Track epsilon history

    def step(self, activation_noise_multiplier=0.0, gradient_noise_multiplier=None, sampling_rate=None, num_steps=1):
        # Ignore activation_noise_multiplier as vanilla accountant only tracks gradient noise
        # Use provided gradient_noise_multiplier or default to the fixed sigma
        sigma = gradient_noise_multiplier if gradient_noise_multiplier is not None else self.fixed_sigma
        q = sampling_rate if sampling_rate is not None else self.q
        
        # Call super().step to update the RDP values
        super().step(sigma, q, num_steps)
        
        # Track epsilon after step
        eps, _ = self.get_privacy_spent(delta=1e-5)  # Using a default delta for tracking
        self._eps_history.append(eps)

class UnifiedGaussianAccountant(ManualPrivacyAccountant):
    """
    Unified Gaussian accountant that tracks privacy loss from both activation noise
    and gradient noise in a single privacy budget calculation. This replaces the
    hybrid Laplace-Gaussian approach with a unified Gaussian-only mechanism.
    """
    def __init__(self, noise_multiplier, sampling_rate, moment_orders=None):
        super().__init__(moment_orders)
        self.initial_sigma = noise_multiplier
        self.q = sampling_rate
        self._eps_history = []  # Track epsilon history

    def step(self, activation_noise_multiplier, gradient_noise_multiplier, sampling_rate=None, num_steps=1):
        """
        Updates the privacy accounting by combining privacy loss from both:
        1. Gaussian noise added to client activations
        2. Gaussian noise added to client gradients
        
        The privacy loss from both sources in a round is combined and added to the budget.
        
        Args:
            activation_noise_multiplier: Noise scale for client activations
            gradient_noise_multiplier: Noise scale for client gradients 
            sampling_rate: Sampling probability (or use default from init)
            num_steps: Number of steps to account for (default: 1)
        """
        # Use provided sampling_rate or default to initialization value
        q = sampling_rate if sampling_rate is not None else self.q
        
        # Account for activation noise (if applied)
        if activation_noise_multiplier > 0:
            super().step(activation_noise_multiplier, q, num_steps)
            
        # Account for gradient noise (if applied)
        if gradient_noise_multiplier > 0:
            super().step(gradient_noise_multiplier, q, num_steps)
            
        # Track epsilon after both noise additions
        eps, _ = self.get_privacy_spent(delta=1e-5)  # Using a default delta for tracking
        self._eps_history.append(eps)

def get_accountant(mode: str, **kw):
    if mode == "vanilla":
        return VanillaGaussianAccountant(**kw)
    if mode == "adaptive":
        return UnifiedGaussianAccountant(**kw)
    if mode == "unified":
        return UnifiedGaussianAccountant(**kw)
    raise ValueError(f"Unknown accountant mode {mode}") 