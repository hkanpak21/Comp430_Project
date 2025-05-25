# Secure Split Federated Learning (SFL) Framework

## Overview

This project implements a simulation framework for Secure Split Federated Learning (SFL), specifically the SFLV1 variant described by Thapa et al. (2022). It incorporates manually implemented privacy-enhancing mechanisms (Laplacian noise on activations, Gaussian noise on gradients with per-sample clipping) and a manual Moments Accountant for privacy tracking, without relying on external DP libraries like Opacus or dp-accounting. The framework includes both fixed and adaptive differential privacy implementations.

## Features

*   **SFLV1 Simulation:** Implements the core workflow with Client, Main Server, and Fed Server components.
*   **Manual Laplacian Noise:** Adds Laplacian noise to smashed data (activations) at the client-side.
*   **Manual Gaussian Noise:** Applies per-sample gradient clipping and Gaussian noise to client-side model gradients.
*   **Adaptive Differential Privacy:** Implements adaptive clipping thresholds and noise scaling based on model convergence, as described in Fu et al. (2022).
*   **Manual Moments Accountant:** Tracks (ε, δ)-differential privacy budget for the Gaussian noise mechanism, including support for adaptive noise scales.
*   **Model Splitting:** Allows splitting PyTorch models at a specified layer.
*   **Federated Averaging:** Basic implementation for model aggregation.
*   **Configurable:** Key parameters managed via YAML configuration files.
*   **Evaluation:** Reports test accuracy and privacy budget.

## Repository Structure

```
Comp430_Project/
├── configs/               # Configuration files for experiments
│   ├── default.yaml       # Default configuration (minimal DP noise)
│   ├── fixed_dp.yaml      # Configuration for fixed DP noise
│   ├── adaptive_dp.yaml   # Configuration for adaptive DP noise
│   └── ...                # Model-specific configurations
├── data/                  # Placeholder for downloaded datasets
├── experiments/           # Main training and evaluation scripts
│   └── train_secure_sfl.py # Main training script for SFLV1
├── src/                   # Source code
│   ├── datasets/          # Data loading and partitioning
│   │   └── data_loader.py # Functions for loading and partitioning datasets
│   ├── dp/                # Differential privacy components
│   │   ├── noise_utils.py # Manual noise functions (Laplacian, Gaussian)
│   │   └── privacy_accountant.py # Manual privacy accounting
│   ├── models/            # Model definitions and splitting logic
│   │   ├── simple_cnn.py  # CNN model implementation
│   │   ├── simple_dnn.py  # DNN model implementation
│   │   └── split_utils.py # Utilities for splitting models
│   ├── sfl/               # SFL core components
│   │   ├── aggregation.py # Federated Averaging implementation
│   │   ├── client.py      # SFLClient implementation
│   │   ├── fed_server.py  # Federation Server implementation
│   │   └── main_server.py # Main Server implementation
│   └── utils/             # Utility functions
│       └── config_parser.py # Configuration parsing
├── Papers/                # Reference papers and materials
├── experiment_logs/       # Logs from experiments
├── experiment_results/    # Results from experiments
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

Experiment parameters are defined in YAML files within the `configs/` directory. The repository includes several configuration templates:

* `default.yaml`: Basic configuration with minimal privacy noise.
* `fixed_dp.yaml`: Configuration for fixed differential privacy parameters.
* `adaptive_dp.yaml`: Configuration for adaptive differential privacy.
* Model-specific configurations: `cnn_fixed_dp.yaml`, `dnn_adaptive_dp.yaml`, etc.

### Configuration Parameters

* **Basic Parameters:**
  * `dataset`: Dataset name (e.g., 'MNIST').
  * `model`: Model name (e.g., 'SimpleCNN', 'SimpleDNN').
  * `num_clients`: Total number of clients.
  * `batch_size`: Training batch size.
  * `num_rounds`: Total communication rounds.
  * `lr`: Learning rate.
  * `optimizer`: Optimizer type (e.g., 'SGD', 'Adam').
  * `cut_layer`: Index of the layer where the model is split.

* **Fixed DP Parameters:**
  * `laplacian_sensitivity`: Sensitivity for Laplacian noise.
  * `epsilon_prime`: Epsilon' for Laplacian noise.
  * `clip_norm`: L2 norm bound for gradient clipping.
  * `noise_multiplier`: Noise multiplier for Gaussian noise.
  * `delta`: Target delta for privacy accounting.

* **Adaptive DP Parameters:**
  * `adaptive_clipping_factor`: Factor for adaptive clipping threshold.
  * `initial_sigma`: Initial noise scale for adaptive DP.
  * `adaptive_noise_decay_factor`: Factor to decrease noise scale.
  * `noise_decay_patience`: Rounds of loss decrease before reducing noise.
  * `validation_set_ratio`: Ratio of data for validation set.

## Running Experiments

From the repository root directory:

```bash
# Basic run with default configuration
python experiments/train_secure_sfl.py --config configs/default.yaml

# Run with fixed differential privacy
python experiments/train_secure_sfl.py --config configs/fixed_dp.yaml

# Run with adaptive differential privacy
python experiments/train_secure_sfl.py --config configs/adaptive_dp.yaml

# Model-specific configurations
python experiments/train_secure_sfl.py --config configs/cnn_adaptive_dp.yaml
```

## Output

The training script will:
1. Log training progress for each round
2. Report current privacy budget (ε, δ) at specified intervals
3. For adaptive DP, show the evolution of noise scale (σ)
4. Display final test accuracy of the globally aggregated model
5. Report the final privacy budget (ε, δ)

## References

* Thapa, C., et al. (2022). "Split Learning for Collaborative Deep Learning with Differential Privacy"
* Fu, A., et al. (2022). "Adap DP-FL: Adaptive Differential Privacy for Federated Learning"
