# Secure Split-FL Test Suite

This directory contains test cases to validate the correctness of the Secure Split-FL implementation.

## Test Cases

1. **Pipeline Test** (`test_pipeline.py`)
   - Verifies that the training pipeline runs successfully
   - Checks that training completes at least one round
   - Confirms that accuracy metrics are properly logged

2. **DP Budget Test** (`test_dp_budget.py`)
   - `test_fixed_dp`: Ensures fixed DP budget doesn't exceed maximum epsilon (8.0)
   - `test_adaptive_sigma`: Verifies adaptive noise mechanism reduces sigma over time

3. **FedAvg Equivalence Test** (`test_fedavg_equiv.py`)
   - Tests that 1-client SFL is equivalent to centralized training within a small error margin
   - Validates correctness of aggregation mechanism

4. **Utility Test** (`test_utility.py`)
   - Ensures the model achieves minimum accuracy threshold specified in config

5. **Known Configuration Test** (`test_known_config.py`)
   - Tests with known working configuration (cnn_adaptive_dp.yaml)
   - Validates expected performance from previous runs

## Running Tests

Run all tests with:
```bash
pytest -q experiments/tests
```

Run a specific test with:
```bash
pytest -q experiments/tests/test_pipeline.py
```

## Test Results

Test artifacts are stored in `experiments/out/{config_name}__{uuid}/`
- `metrics.json`: Contains final metrics and parameters
- `stdout.txt`: Captures training output
- `stderr.txt`: Captures any errors
