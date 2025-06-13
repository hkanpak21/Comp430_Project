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

## Repository Structure

```
Comp430_Project/
├── configs/               # YAML configuration files for experiments
├── data/                  # Placeholder for datasets
├── experiments/           # Main training scripts (e.g., train_secure_sfl.py)
├── src/                   # Source code
│   ├── datasets/          # Data loading
│   ├── dp/                # Differential privacy (noise, accountant)
│   ├── models/            # Model definitions, splitting logic
│   ├── sfl/               # SFL components (client, servers)
│   └── utils/             # Utility functions
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

Experiments are configured using YAML files in the `configs/` directory. The framework now includes the following configuration types:

1. **Client Scaling:** 3, 5, 10, and 20 clients (`mnist_clients{n}_vanilla_dp.yaml`)
2. **Split Variations:** 
   - Federated Learning (`mnist_cut_layer_fl.yaml`)
   - Centralized Learning (`mnist_cut_layer_central.yaml`)
   - Split learning with various cut layers
3. **DP Mechanisms:**
   - Vanilla DP (fixed privacy parameters)
   - Adaptive DP (adjusts noise based on validation loss)
4. **Datasets:**
   - MNIST (`mnist_*.yaml`)
   - Breast Cancer Wisconsin (`bcw_*.yaml`)

## Running Experiments

Execute experiments from the repository root:

```bash
# MNIST with 5 clients and vanilla DP
python experiments/train_secure_sfl.py --config configs/mnist_clients5_vanilla_dp.yaml --run_id mnist_clients5_vanilla

# MNIST with 5 clients and adaptive DP
python experiments/train_secure_sfl.py --config configs/mnist_clients5_adaptive_dp.yaml --run_id mnist_clients5_adaptive

# Breast Cancer Wisconsin with 5 clients
python experiments/train_secure_sfl.py --config configs/bcw_clients5_vanilla_dp.yaml --run_id bcw_clients5_vanilla

# Federated Learning simulation (cut at last layer)
python experiments/train_secure_sfl.py --config configs/mnist_cut_layer_fl.yaml --run_id mnist_fl_sim
```

The `--run_id` parameter specifies an output directory for metrics under `experiments/out/`.

## Output

The framework generates detailed metrics in JSON format, including:
- Final test accuracy
- Total training time
- Per-round metrics (accuracy, privacy budget, noise scales, etc.)
- Privacy parameters
- Full configuration

The metrics are stored in `experiments/out/{run_id}/metrics.json`.

## References

*   Thapa, C., et al. (2022). "Split Learning for Collaborative Deep Learning with Differential Privacy"
*   Fu, A., et al. (2022). "Adap DP-FL: Adaptive Differential Privacy for Federated Learning"
