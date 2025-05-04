#!/usr/bin/env python3
import math
from typing import List, Union

# Helper functions (avoiding scipy)

def _log_comb(n: int, k: int) -> float:
    """Computes log(C(n, k)) using log gamma functions."""
    if k < 0 or k > n:
        return -float('inf') # Log of zero
    # Using math.lgamma which computes log(Gamma(x)) = log((x-1)!)
    # log(n! / (k! * (n-k)!)) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

def _log_add_exp(log_a: float, log_b: float) -> float:
    """Computes log(exp(log_a) + exp(log_b)) robustly."""
    if log_a == -float('inf'):
        return log_b
    if log_b == -float('inf'):
        return log_a
    if log_a > log_b:
        return log_a + math.log1p(math.exp(log_b - log_a))
    else:
        return log_b + math.log1p(math.exp(log_a - log_b))

def _compute_rdp_epsilon_step(q: float, noise_multiplier: float, alpha: int) -> float:
    """
    Computes the Renyi Differential Privacy (RDP) epsilon for a single step
    of the sampled Gaussian mechanism.

    Based on Mironov (2017) "Renyi Differential Privacy" and Wang et al.
    (Subsampled Renyi DP). This computes epsilon(alpha) for one step.

    Args:
        q: Sampling rate (probability).
        noise_multiplier: The ratio of std deviation to the clipping norm (sigma/C).
        alpha: The RDP order (integer > 1).

    Returns:
        The RDP epsilon for order alpha for a single step. Returns infinity if
        noise_multiplier is zero and q > 0.
    """
    if q == 0:
        return 0.0 # No privacy cost if not sampled
    if q == 1.0:
        # Standard (non-sampled) Gaussian mechanism RDP
        if noise_multiplier == 0:
             return float('inf') # Infinite privacy cost with zero noise
        # RDP is alpha / (2 * sigma^2) where sigma is noise_multiplier
        return alpha / (2.0 * noise_multiplier**2)
    if noise_multiplier == 0:
         return float('inf') # Infinite privacy cost if noise is zero and sampled

    sigma_squared = noise_multiplier**2

    # Compute the sum using log-sum-exp trick for numerical stability
    log_sum_exp = -float('inf')
    log_q = math.log(q)
    log_1_minus_q = math.log1p(-q) # More accurate for small q

    for k in range(alpha + 1):
        log_comb_term = _log_comb(alpha, k)
        if log_comb_term == -float('inf'):
            continue # Skip k=0 or k=alpha if q=1 or q=0 handled above? Check logic. C(a,0)=1, C(a,a)=1.

        # Term involving probabilities: k * log(q) + (alpha - k) * log(1-q)
        log_prob_term = k * log_q + (alpha - k) * log_1_minus_q

        # Term involving the RDP of non-sampled mechanism at order k
        # We need exp((k-1) * rdp_epsilon(k)) = exp((k-1) * k / (2 * sigma^2))
        # Handle k=0 and k=1 where the exponent term is 0 -> exp(0) = 1 -> log(1) = 0
        log_exp_term = 0.0
        if k > 1:
            log_exp_term = (k - 1.0) * k / (2.0 * sigma_squared)

        # Combine terms in log space: log(C(a,k) * q^k * (1-q)^(a-k) * exp(...))
        current_term_log = log_comb_term + log_prob_term + log_exp_term

        # Add to the total sum using log-add-exp
        log_sum_exp = _log_add_exp(log_sum_exp, current_term_log)

    # Final RDP epsilon is log(sum) / (alpha - 1)
    rdp_epsilon = log_sum_exp / (alpha - 1.0)

    return rdp_epsilon


class ManualPrivacyAccountant:
    """
    Manually implemented Moments Accountant (based on Renyi DP) to track
    cumulative privacy cost (epsilon, delta) for the Gaussian Noise mechanism
    (Noise Mechanism 2: per-sample clipping + Gaussian noise) over steps.

    Does NOT use external libraries like dp-accounting or Opacus.
    Focuses solely on the privacy cost from Mechanism 2.
    """
    def __init__(self,
                 moment_orders: List[Union[int, float]] = None):
        """
        Initializes the accountant.

        Args:
            moment_orders: A list of RDP orders (alpha values > 1) to track.
                           If None, uses a default set.
        """
        # Use a default set of orders if none provided (common practice)
        if moment_orders is None:
            moment_orders = list(range(2, 33)) + [40.0, 48.0, 56.0, 64.0]
            # Ensure orders are floats for calculations if needed later, although
            # the current RDP formula assumes integer alphas for combinations.
            # Let's keep them as provided or default integers/floats.
            # For the current formula, ensure they are integers > 1
            moment_orders = [int(a) for a in moment_orders if isinstance(a, (int, float)) and a > 1]
            moment_orders = sorted(list(set(moment_orders))) # Unique & sorted

        if not moment_orders or any(alpha <= 1 for alpha in moment_orders):
            raise ValueError("Moment orders (alphas) must be > 1.")

        self.moment_orders = moment_orders
        # Store total accumulated RDP epsilon for each order alpha
        self._total_rdp_epsilons = {alpha: 0.0 for alpha in self.moment_orders}
        self._steps = 0 # Track total number of steps taken

    def step(self, noise_multiplier: float, sampling_rate: float, num_steps: int = 1):
        """
        Records the privacy cost of applying the sampled Gaussian mechanism
        for a number of steps.

        Args:
            noise_multiplier: The noise multiplier (sigma/C) used in the step(s).
            sampling_rate: The sampling rate (q = batch_size / dataset_size) used.
            num_steps: The number of steps taken with these parameters (default: 1).
        """
        if noise_multiplier < 0:
            raise ValueError("Noise multiplier cannot be negative.")
        if not (0 <= sampling_rate <= 1):
            raise ValueError("Sampling rate must be between 0 and 1.")
        if num_steps <= 0:
            return # No steps taken

        for alpha in self.moment_orders:
            # Calculate RDP epsilon for a *single* step with these params
            # Ensure alpha is int for _compute_rdp_epsilon_step as implemented
            rdp_epsilon_step = _compute_rdp_epsilon_step(sampling_rate, noise_multiplier, int(alpha))

            # Accumulate the total RDP epsilon for this order
            self._total_rdp_epsilons[alpha] += num_steps * rdp_epsilon_step

        self._steps += num_steps

    def get_privacy_spent(self, delta: float) -> tuple[float, float]:
        """
        Computes the (epsilon, delta)-DP guarantee for the accumulated
        privacy cost.

        Args:
            delta: The target delta. Must be > 0.

        Returns:
            A tuple (epsilon, delta) where epsilon is the smallest epsilon
            found for the given delta across all tracked RDP orders.
            Returns (inf, delta) if delta <= 0 or if epsilon is infinite for all orders.
        """
        if delta <= 0:
            print("Warning: Target delta must be positive.")
            # Or raise ValueError("Target delta must be positive.")
            return float('inf'), delta

        min_epsilon = float('inf')

        for alpha in self.moment_orders:
            total_rdp_epsilon = self._total_rdp_epsilons[alpha]

            if total_rdp_epsilon == float('inf'):
                continue # This alpha gives infinite epsilon

            # Formula to convert RDP epsilon(alpha) to (epsilon, delta)-DP:
            # epsilon = RDP_epsilon(alpha) - log(delta) / (alpha - 1)
            epsilon = total_rdp_epsilon - (math.log(delta) / (alpha - 1.0))

            # Ensure epsilon is not negative (can happen with large delta/small RDP)
            # Privacy guarantees cannot have negative epsilon.
            # Epsilon=0 means no privacy loss beyond delta.
            epsilon = max(0.0, epsilon)

            min_epsilon = min(min_epsilon, epsilon)

        return min_epsilon, delta

    @property
    def total_steps(self):
        return self._steps 