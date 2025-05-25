from .utils import run_experiment

def test_forward_backward():
    """Tests that the pipeline works by running a basic experiment."""
    m = run_experiment("configs/default.yaml")
    # —— Success criteria ——
    assert 0.0 < m["final_test_acc"] < 1.0,  "accuracy not logged"
    assert m["rounds"] >= 1,                 "training never started"
