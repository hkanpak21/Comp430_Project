# Secure Split Federated Learning (SFL) Framework

## Overview

This project implements a simulation framework for Secure Split Federated Learning (SFLV1), as described by Thapa et al. (2022). It features a unified Gaussian differential privacy mechanism for both activations and gradients, and a comprehensive experimental suite for evaluating various configurations.

## Key Features

*   **SFLV1 Simulation:** Client, Main Server, and Fed Server components.
*   **Unified Gaussian DP Mechanism:**
    *   Gaussian noise for both activations and gradients with appropriate clipping.
    *   Unified privacy accounting via Rényi Differential Privacy.
*   **Adaptive Differential Privacy:** Adjusts clipping and noise based on model convergence.
*   **Multiple Datasets:** MNIST and Breast Cancer Wisconsin (BCW).
*   **Experimental Suite:** Varied client counts, split strategies, and privacy mechanisms.
*   **Model Splitting:** Supports splitting PyTorch models at specified layers.
*   **Federated Averaging:** For model aggregation.
*   **Configurable:** Uses YAML files for experiment parameters.
*   **Detailed Logging:** Comprehensive per-round metrics in JSON format.
*   **Batch Experiment Runner:** Run multiple experiments with a single command.

## Repository Structure

```
Comp430_Project/
├── configs/               # YAML configuration files for experiments
├── data/                  # Placeholder for datasets
├── experiments/           # Main training scripts (e.g., train_secure_sfl.py)
├── results/               # Default output directory for batch experiment results
├── src/                   # Source code
│   ├── datasets/          # Data loading
│   ├── dp/                # Differential privacy (noise, accountant)
│   ├── models/            # Model definitions, splitting logic
│   ├── sfl/               # SFL components (client, servers)
│   └── utils/             # Utility functions
├── run_experiments.py     # Batch experiment runner script
└── requirements.txt       # Python dependencies
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/Comp430_Project.git
    cd Comp430_Project
    ```
2.  **Create a Python environment:** (Recommended)
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

Experiments are configured using YAML files in the `configs/` directory. The framework includes the following configuration types:

1. **Client Scaling:** 3, 5, 10, 20, and 50 clients (`mnist_clients{n}_vanilla_dp.yaml` and `mnist_clients{n}_adaptive_dp.yaml`)
2. **Split Variations:** 
   - Federated Learning (`mnist_cut_layer_fl.yaml`)
   - Centralized Learning (`mnist_cut_layer_central.yaml`)
   - Split learning with various cut layers (2-9)
3. **DP Mechanisms:**
   - Vanilla DP (fixed privacy parameters)
   - Adaptive DP (adjusts noise based on validation loss)
   - Fixed Noise (constant noise without adaptation)
   - No Noise (baseline without DP)
4. **Datasets:**
   - MNIST (`mnist_*.yaml`)
   - Breast Cancer Wisconsin (`bcw_*.yaml`)
5. **Data Distribution:**
   - IID (default)
   - Non-IID with various concentration parameters

## Running Individual Experiments

Execute experiments from the repository root:

```bash
# MNIST with 5 clients and vanilla DP
python experiments/train_secure_sfl.py --config configs/mnist_clients5_vanilla_dp.yaml --run_id mnist_clients5_vanilla

# MNIST with 5 clients and adaptive DP
python experiments/train_secure_sfl.py --config configs/mnist_clients5_adaptive_dp.yaml --run_id mnist_clients5_adaptive

# Breast Cancer Wisconsin with 5 clients
python experiments/train_secure_sfl.py --config configs/bcw_clients5_vanilla_dp.yaml --run_id bcw_clients5_vanilla

# Split at layer 3 with adaptive DP
python experiments/train_secure_sfl.py --config configs/mnist_clients5_cut_layer3_adaptive_dp.yaml --run_id mnist_cut3_adaptive
```

The `--run_id` parameter specifies an output directory for metrics under `experiments/out/`.

## Batch Experiment Runner

For running multiple experiments sequentially, use the `run_experiments.py` script:

```bash
# Run all configurations
python run_experiments.py --output_dir results

# Run only MNIST experiments with 5 clients
python run_experiments.py --output_dir results/mnist_clients5 --filter mnist_clients5

# Run only cut layer experiments
python run_experiments.py --output_dir results/cut_layers --filter cut_layer

# Set timeout for long-running experiments (in seconds)
python run_experiments.py --timeout 3600
```

The batch runner:
- Executes each experiment sequentially
- Collects metrics and logs in an organized directory structure
- Generates a summary report with key metrics
- Handles timeouts and errors gracefully

## Output

The framework generates detailed metrics in JSON format, including:
- Final test accuracy
- Total training time
- Per-round metrics (accuracy, privacy budget, noise scales, etc.)
- Privacy parameters
- Full configuration

Individual experiment metrics are stored in `experiments/out/{run_id}/metrics.json`.

Batch experiment results are organized in the specified output directory:
```
results/
├── {config_name1}/
│   ├── metrics.json    # Experiment metrics
│   ├── stdout.log      # Standard output log
│   └── stderr.log      # Error log
├── {config_name2}/
│   └── ...
└── summary.json        # Summary of all experiments
```

## Model Split Layers

The framework supports splitting models at various layers to evaluate the impact on performance and privacy. Available cut layers for the MNIST CNN model include:

- Layers 2-9: Covering early, middle, and late splits in the network
- Each layer can be combined with either vanilla or adaptive DP

## References

*   Thapa, C., et al. (2022). "Split Learning for Collaborative Deep Learning with Differential Privacy"
*   Fu, A., et al. (2022). "Adap DP-FL: Adaptive Differential Privacy for Federated Learning"
