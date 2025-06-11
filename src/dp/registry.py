from .privacy_accountant import ManualPrivacyAccountant

class LaplaceAccumulator:
    """Simple pure-DP counter for Laplace ε's."""
    def __init__(self):
        self.eps_sum = 0.0
        self._eps_history = []  # Track epsilon history

    def step(self, eps: float):
        assert eps >= 0
        self.eps_sum += eps
        self._eps_history.append(self.eps_sum)

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
        # Call super().step to update the RDP values
        super().step(self.fixed_sigma, self.q, num_steps)
        
        # Track epsilon after step
        eps, _ = self.get_privacy_spent(delta=1e-5)  # Using a default delta for tracking
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

class HybridAccountant:
    """
    Tracks:
      1) pure-DP ε from Laplace steps,
      2) RDP from Gaussian steps via ManualPrivacyAccountant.
    """
    def __init__(self, noise_multiplier, sampling_rate, moment_orders=None):
        # for Gaussian RDP
        self.gauss_acc = ManualPrivacyAccountant(moment_orders)
        self.fixed_sigma = noise_multiplier
        self.q = sampling_rate
        self._eps_history = []  # Track total epsilon history

        # for Laplace pure-DP
        self.laplace_acc = LaplaceAccumulator()

    def laplace_step(self, epsilon_prime: float):
        """Call every time you inject Laplace(…,ε′)."""
        self.laplace_acc.step(epsilon_prime)
        # Update the combined history
        self._update_history()

    def gaussian_step(self, noise_multiplier=None, num_steps: int = 1):
        """Call every time you inject Gaussian noise on gradients."""
        # uses the *fixed* sigma for all steps (unless adaptive is provided)
        sigma = noise_multiplier if noise_multiplier is not None else self.fixed_sigma
        self.gauss_acc.step(sigma, self.q, num_steps)
        # Update the combined history
        self._update_history()

    def _update_history(self):
        """Update the combined epsilon history."""
        eps_gauss, _ = self.gauss_acc.get_privacy_spent(delta=1e-5)
        eps_lap = self.laplace_acc.eps_sum
        self._eps_history.append(eps_lap + eps_gauss)

    def get_privacy_spent(self, delta: float):
        """
        Returns the composed (ε,δ):
          ε = ε_laplace + ε_gauss
          δ = δ  (pure-DP from Laplace contributes no δ)
        """
        eps_gauss, _ = self.gauss_acc.get_privacy_spent(delta)
        eps_lap = self.laplace_acc.eps_sum
        return eps_lap + eps_gauss, delta
        
    @property
    def epsilon_laplace(self):
        """Get the current Laplace privacy cost."""
        return self.laplace_acc.eps_sum
        
    @property
    def epsilon_gaussian(self):
        """Get the current Gaussian privacy cost."""
        eps_gauss, _ = self.gauss_acc.get_privacy_spent(delta=1e-5)
        return eps_gauss
        
    @property
    def total_steps(self):
        """Get the total number of Gaussian steps."""
        return self.gauss_acc.total_steps

def get_accountant(mode: str, **kw):
    if mode == "vanilla":
        return VanillaGaussianAccountant(**kw)
    if mode == "adaptive":
        return AdaptiveGaussianAccountant(**kw)
    if mode == "hybrid":
        return HybridAccountant(**kw)
    raise ValueError(f"Unknown accountant mode {mode}") 