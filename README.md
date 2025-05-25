# Secure Split Federated Learning (SFL) Framework

## Overview

This project implements a simulation framework for Secure Split Federated Learning (SFLV1), as described by Thapa et al. (2022). It features manually implemented privacy-enhancing techniques (Laplacian noise on activations, Gaussian noise on gradients with per-sample clipping) and a manual Moments Accountant for privacy tracking. The framework supports both fixed and adaptive differential privacy (inspired by Fu et al., 2022) without external DP libraries.

## Key Features

*   **SFLV1 Simulation:** Client, Main Server, and Fed Server components.
*   **Manual DP Mechanisms:**
    *   Laplacian noise for activations.
    *   Gaussian noise with per-sample gradient clipping for gradients.
*   **Adaptive Differential Privacy:** Adjusts clipping and noise based on model convergence.
*   **Manual Moments Accountant:** Tracks (ε, δ)-DP budget.
*   **Model Splitting:** Supports splitting PyTorch models at specified layers.
*   **Federated Averaging:** For model aggregation.
*   **Configurable:** Uses YAML files for experiment parameters.
*   **Evaluation:** Reports test accuracy and privacy budget.

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
├── Papers/                # Reference papers (archived, in .gitignore)
├── experiment_logs/       # Experiment logs
├── experiment_results/    # Experiment results
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

Experiments are configured using YAML files in the `configs/` directory (e.g., `default.yaml`, `fixed_dp.yaml`, `adaptive_dp.yaml`). Key parameters include:

*   **Basic:** Dataset, model, client/round numbers, learning rate, optimizer, cut layer.
*   **Fixed DP:** Laplacian sensitivity/epsilon, gradient clip norm, noise multiplier, delta.
*   **Adaptive DP:** Clipping factor, initial sigma, noise decay parameters, validation ratio.

Refer to the example configuration files in `configs/` for detailed parameter lists and descriptions.

## Running Experiments

Execute experiments from the repository root:

```bash
# Default configuration
python experiments/train_secure_sfl.py --config configs/default.yaml

# Fixed differential privacy
python experiments/train_secure_sfl.py --config configs/fixed_dp.yaml

# Adaptive differential privacy
python experiments/train_secure_sfl.py --config configs/adaptive_dp.yaml

# Example with a specific model and adaptive DP
python experiments/train_secure_sfl.py --config configs/cnn_adaptive_dp.yaml
```

## Output

The script logs training progress, privacy budget (ε, δ) evolution (including noise scale σ for adaptive DP), final test accuracy, and the final privacy budget.

## References

*   Thapa, C., et al. (2022). "Split Learning for Collaborative Deep Learning with Differential Privacy"
*   Fu, A., et al. (2022). "Adap DP-FL: Adaptive Differential Privacy for Federated Learning"
