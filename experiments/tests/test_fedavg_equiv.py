import torch
import math
from .utils import run_experiment

THRESH = 1e-3
def test_single_client_equivalence():
    """Tests that 1-client SFL is equivalent to centralized training."""
    m_split = run_experiment("configs/one_client.yaml")
    m_central = run_experiment("configs/central.yaml")
    diff = abs(m_split["final_test_acc"] - m_central["final_test_acc"])
    assert diff < THRESH, f"Split = {m_split}, Central = {m_central}"
