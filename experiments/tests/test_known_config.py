from .utils import run_experiment

def test_known_working_config():
    """
    Tests the known working configuration (cnn_adaptive_dp.yaml) that previously 
    achieved 75.96% accuracy with DP noise disabled.
    """
    m = run_experiment("configs/cnn_adaptive_dp.yaml")
    
    # We know from previous runs that this config works well
    assert m["final_test_acc"] > 0.70, "Expected accuracy of at least 70% with this config"
    
    # Confirm DP settings are as expected
    assert m["sigma_init"] == 0.0 or m["sigma"] == 0.0, "DP noise should be disabled in this test"
