import sys
import os
import pytest

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.dp.registry import get_accountant

@pytest.mark.parametrize("mode", ["vanilla", "adaptive"])
def test_budget_progression(mode):
    acc = get_accountant(mode,
                         noise_multiplier=1.2,
                         sampling_rate=0.01)
    
    if mode == "vanilla":
        acc.step(num_steps=100)
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

def test_invalid_accountant_mode():
    with pytest.raises(ValueError):
        get_accountant("invalid_mode", noise_multiplier=1.0, sampling_rate=0.01) 