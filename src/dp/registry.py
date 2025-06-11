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

    def step(self, num_steps: int = 1):
        # First call super().step to update the RDP values
        super().step(self.fixed_sigma, self.q, num_steps)
        
        # Then get the current privacy spent and append to history
        # This should only append once per call
        eps, _ = self.get_privacy_spent(delta=1e-5)
        # Clear any existing history from parent class
        if hasattr(super(), '_eps_history'):
            super()._eps_history = []
        # Append to our history
        self._eps_history.append(eps)

class AdaptiveGaussianAccountant(ManualPrivacyAccountant):
    """
    Extends ManualPrivacyAccountant to store initialization parameters
    but allows sigma to be changed over time.
    """
    def __init__(self, noise_multiplier, sampling_rate, moment_orders=None):
        super().__init__(moment_orders)
        self.initial_sigma = noise_multiplier
        self.q = sampling_rate
        self._eps_history = []  # Track epsilon history

    def step(self, noise_multiplier, sampling_rate=None, num_steps=1):
        # Use provided sampling_rate or default to initialization value
        q = sampling_rate if sampling_rate is not None else self.q
        super().step(noise_multiplier, q, num_steps)
        # Track epsilon after step
        eps, _ = self.get_privacy_spent(delta=1e-5)  # Using a default delta for tracking
        self._eps_history.append(eps)

def get_accountant(mode: str, **kw):
    if mode == "vanilla":
        return VanillaGaussianAccountant(**kw)
    if mode == "adaptive":
        return AdaptiveGaussianAccountant(**kw)
    raise ValueError(f"Unknown accountant mode {mode}") 