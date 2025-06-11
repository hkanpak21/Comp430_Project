import sys
import os
import pytest

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.dp.registry import get_accountant

@pytest.mark.parametrize("mode", ["vanilla", "adaptive", "hybrid"])
def test_budget_progression(mode):
    acc = get_accountant(mode,
                         noise_multiplier=1.2,
                         sampling_rate=0.01)
    
    if mode == "vanilla":
        acc.step(num_steps=100)
    elif mode == "hybrid":
        # For hybrid, we test both Laplace and Gaussian steps
        acc.laplace_step(epsilon_prime=0.1)
        acc.gaussian_step(num_steps=100)
    else:
        acc.step(noise_multiplier=1.2, num_steps=100)
        
    eps, _ = acc.get_privacy_spent(delta=1e-5)
    assert eps < 10      # smoke-level bound

@pytest.mark.parametrize("mode", ["vanilla", "adaptive"])
def test_accountant_types(mode):
    acc = get_accountant(mode,
                         noise_multiplier=1.0,
                         sampling_rate=0.01)
    
    # Clear any existing history that might be initialized
    acc._eps_history = []
    
    if mode == "vanilla":
        # VanillaGaussianAccountant should have fixed sigma
        assert hasattr(acc, 'fixed_sigma')
        assert acc.fixed_sigma == 1.0
        # Test step method
        acc.step(num_steps=10)
        assert len(acc._eps_history) == 1
    else:
        # Adaptive accountant
        assert hasattr(acc, 'initial_sigma')
        # Test step method with different sigmas
        acc.step(noise_multiplier=1.0, num_steps=5)
        acc.step(noise_multiplier=0.8, num_steps=5)
        assert len(acc._eps_history) == 2

def test_hybrid_accountant():
    """Test specific functionality of the hybrid accountant."""
    acc = get_accountant("hybrid",
                         noise_multiplier=1.0,
                         sampling_rate=0.01)
    
    # Clear any existing history
    acc._eps_history = []
    
    # Test Laplace steps
    acc.laplace_step(epsilon_prime=0.1)
    assert acc.epsilon_laplace == 0.1
    
    # Test Gaussian steps
    acc.gaussian_step(num_steps=10)
    assert acc.total_steps == 10
    
    # Test combined privacy calculation
    eps, delta = acc.get_privacy_spent(delta=1e-5)
    assert eps > 0.1  # Should be greater than just the Laplace component
    assert delta == 1e-5
    
    # Check that epsilon_gaussian property works
    assert acc.epsilon_gaussian > 0
    assert abs(eps - (acc.epsilon_laplace + acc.epsilon_gaussian)) < 1e-10
    
    # Test history tracking
    assert len(acc._eps_history) == 2  # One entry per step

def test_invalid_accountant_mode():
    with pytest.raises(ValueError):
        get_accountant("invalid_mode", noise_multiplier=1.0, sampling_rate=0.01) 