# Secure Split Federated Learning (SFL) Framework

## Overview

This project implements a simulation framework for Secure Split Federated Learning (SFL), specifically the SFLV1 variant described by Thapa et al. (2022). It incorporates manually implemented privacy-enhancing mechanisms (Laplacian noise on activations, Gaussian noise on gradients with per-sample clipping) and a manual Moments Accountant for privacy tracking, without relying on external DP libraries like Opacus or dp-accounting.

## Features

*   **SFLV1 Simulation:** Implements the core workflow with Client, Main Server, and Fed Server components.
*   **Manual Laplacian Noise:** Adds Laplacian noise to smashed data (activations) at the client-side.
*   **Manual Gaussian Noise:** Applies per-sample gradient clipping and Gaussian noise to client-side model gradients.
*   **Manual Moments Accountant:** Tracks (ε, δ)-differential privacy budget for the Gaussian noise mechanism.
*   **Model Splitting:** Allows splitting PyTorch models at a specified layer.
*   **Federated Averaging:** Basic implementation for model aggregation.
*   **Configurable:** Key parameters managed via YAML configuration files.
*   **Evaluation:** Reports test accuracy and privacy budget.

## File Structure

```
Comp430_Project/
├── .git/
├── .gitignore
├── configs/             # Configuration files (e.g., default.yaml)
├── data/                # Placeholder for downloaded datasets
├── datasets/            # Data loading and partitioning (from src)
│   ├── __init__.py
│   └── data_loader.py
├── dp/                  # Differential privacy components (from src)
│   ├── __init__.py
│   ├── noise_utils.py
│   └── privacy_accountant.py
├── experiments/         # Main training and evaluation scripts
│   ├── __init__.py
│   └── train_secure_sfl.py
├── Instructions.md
├── models/              # Model definitions and splitting logic (from src)
│   ├── __init__.py
│   ├── simple_cnn.py
│   └── split_utils.py
├── Papers/
├── README.md
├── requirements.txt
├── sfl/                 # SFL core components (from src)
│   ├── __init__.py
│   ├── aggregation.py
│   ├── client.py
│   ├── fed_server.py
│   └── main_server.py
├── tests/               # Unit tests (Optional but Recommended)
└── utils/               # Utility functions (from src)
    ├── __init__.py
    └── config_parser.py
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
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

Experiment parameters are defined in YAML files within the `configs/` directory (e.g., `configs/default.yaml`). Key parameters include:

*   `dataset`: Dataset name (e.g., 'MNIST').
*   `model`: Model name (e.g., 'SimpleCNN').
*   `num_clients`: Total number of clients.
*   `batch_size`: Training batch size.
*   `num_rounds`: Total communication rounds.
*   `lr`: Learning rate.
*   `optimizer`: Optimizer type (e.g., 'SGD', 'Adam').
*   `cut_layer`: Index of the layer where the model is split.
*   `dp_noise`: Settings for differential privacy:
    *   `laplacian_sensitivity`: Sensitivity (Delta_p,q) for Laplacian noise.
    *   `epsilon_prime`: Epsilon' for Laplacian noise.
    *   `clip_norm`: L2 norm bound for gradient clipping.
    *   `noise_multiplier`: Noise multiplier for Gaussian noise.
    *   `delta`: Target delta for privacy accounting.
*   `seed`: Random seed for reproducibility.

Modify the `configs/default.yaml` file or create new configuration files for different experiments.

## Running Experiments

Ensure you are in the `Comp430_Project` root directory.

To run the main training script:

```bash
# Option 1: Run directly (should work with updated imports)
python experiments/train_secure_sfl.py --config configs/default.yaml

# Option 2: Run as a module (might still be needed depending on environment)
# python -m experiments.train_secure_sfl --config configs/default.yaml 
```

You can specify a different configuration file using the `--config` argument.

## Expected Output

The script will log training progress (loss per round) and output the final test accuracy of the globally aggregated model and the computed privacy budget (ε, δ) based on the manual Moments Accountant. 