import torch
import sys
import os
import pytest

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.dp.privacy_accountant import _compute_rdp_epsilon_step

def test_rdp_empirical():
    torch.manual_seed(0)
    n = 1_000
    v1 = torch.randn(n)
    v2 = v1.clone(); v2[0] += 1.0    # neighbour via one sample shift
    sigma = 2.0
    z = torch.randn_like(v1)*sigma

    mech1 = v1 + z
    mech2 = v2 + z
    kl = (torch.log(torch.exp(-(mech1-v1)**2/(2*sigma**2))) -
          torch.log(torch.exp(-(mech1-v2)**2/(2*sigma**2)))).mean()

    eps_theory = _compute_rdp_epsilon_step(q=1/n,
                                           noise_multiplier=sigma,
                                           alpha=2)
    assert kl <= eps_theory + 0.05 