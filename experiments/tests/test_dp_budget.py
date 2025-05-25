from .utils import run_experiment

# fixed-DP budget must not blow up
MAX_EPS = 8.0
def test_fixed_dp():
    """Tests that the fixed DP budget does not exceed the maximum epsilon."""
    m = run_experiment("configs/fixed_dp.yaml")
    assert m["epsilon"] <= MAX_EPS, "ε budget exceeded"

# adaptive-DP must strictly *decrease* σ over time
def test_adaptive_sigma():
    """Tests that the adaptive noise mechanism reduces noise over time."""
    m = run_experiment("configs/adaptive_dp.yaml")
    assert m["sigma"] < m["sigma_init"], "σ did not decay"
