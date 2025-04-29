# Product Requirements Document (PRD): Split Federated Learning with Adaptive Differential Privacy

---

## 1. Instructions for Implementation Agent

This document outlines the requirements for creating a Python-based code repository for experimenting with Split Federated Learning (SFL) combined with Adaptive Differential Privacy (DP). Please adhere to the following guidelines:

* Implement the core functionalities as specified.
* Follow the proposed file structure.
* Ensure code is well-commented, modular, and follows standard Python practices (PEP 8).
* Use PyTorch or TensorFlow as the primary ML framework (specify which one is chosen or allow configuration).
* Implement experiment scripts that allow easy configuration of SFL and DP parameters via command-line arguments.
* Refer to the provided documentation (papers) for algorithmic details.
* Focus on clarity and reproducibility of experiments.
* Include basic logging for experiment tracking (e.g., accuracy, loss, privacy budget consumption per round).

---

## 2. Core Functionalities

The repository must implement the following core functionalities:

### 2.1. Split Federated Learning (SFL) Framework

* Implement the basic SFL architecture (Version 2 - without parallel FL training within SFL, as described in the SplitFed paper).
* Client-side model part: Handles initial layers of the NN model.
* Server-side model part: Handles later layers of the NN model.
* Mechanism for transferring smashed data (activations from the cut layer) from clients to the server.
* Mechanism for transferring gradients from the server back to the clients.
* Coordination logic for training rounds involving multiple clients and the server.
* Support for configurable model split points.

### 2.2. Differential Privacy Mechanisms

* **Vanilla DP:** Implement standard DP using Gaussian noise addition to gradients or weights, based on a fixed privacy budget ($\epsilon$, $\delta$) and sensitivity (clipping norm $C$).
* **Adaptive DP:** Implement at least one adaptive DP mechanism. Consider options like:
    * **Adaptive Clipping:** Adjust the gradient clipping norm ($C$) based on gradient statistics during training (e.g., based on median or percentiles of gradient norms). Refer to papers like `Adap DP-FL`.
    * **Adaptive Noise:** Adjust the scale of the noise added based on factors like:
        * Training round/convergence status (e.g., decreasing noise over time). Refer to `Adap DP-FL`.
        * Gradient/parameter importance or magnitude. Refer to `Adaptive Differential Privacy in Federated Learning: A Priority-Based Approach`.
        * Layer-wise adaptivity (applying different clipping/noise per layer). Refer to `AWDP-FL`.
    * **Weight-Based Adaptivity:** Implement mechanisms like those in `AWDP-FL` involving historical gradient analysis and weight calculation for adaptive clipping.
* Privacy accounting mechanism (e.g., Moments Accountant) to track the consumed privacy budget.

### 2.3. Experiment Execution and Configuration

* Create a main training script (e.g., `train_sfl_dp.py`).
* The script should accept command-line arguments to configure:
    * Number of clients
    * Dataset (e.g., MNIST, CIFAR-10, Fashion-MNIST)
    * Model architecture and split point
    * Number of training rounds/epochs
    * Learning rate, batch size, optimizer
    * DP mode: (`none`, `vanilla`, `adaptive_clipping`, `adaptive_noise`, `awdp`, etc.)
    * DP parameters (e.g., target $\epsilon$, $\delta$, initial clipping norm $C$, noise multiplier $\sigma$, adaptivity parameters)
    * Data distribution (IID vs. Non-IID simulation)
* Example command format:
    ```bash
    python train_sfl_dp.py --dataset MNIST --num_clients 10 --dp_mode adaptive_noise --target_epsilon 1.0 --target_delta 1e-5 --epochs 50 --model resnet18 --split_layer layer2
    ```bash
    python train_sfl_dp.py --dataset CIFAR10 --num_clients 20 --dp_mode vanilla --clipping_norm 1.0 --noise_multiplier 0.8 --epochs 100 --model vgg11 --split_layer block3
    ```
* Output results (accuracy, loss, privacy budget) to console and optionally to a log file or results directory.

### 2.4. Evaluation and Correctness Check

* Implement standard evaluation metrics (e.g., accuracy on a global test set).
* The agent should include checks or assertions where possible to verify the correct application of DP noise and clipping according to the chosen parameters and mechanism.
* Compare results against baseline (non-private SFL) to understand the privacy-utility trade-off.

---

## 3. Documentation & References

Refer to the following academic papers for algorithmic details and theoretical background. The agent should prioritize understanding the mechanisms described within these papers for implementation.

* **Core SFL:**
    * `SplitFed.pdf`: Introduces the SplitFed Learning concept. Focus on the SFL v2 architecture.
    * `Wu_poisoning_splitfed.pdf`: Provides context on SFL robustness and implementation details (e.g., ResNet usage).
    * `MISA.pdf`: Discusses SFL vulnerabilities, useful for understanding the architecture.
* **Adaptive DP in FL (Adapt concepts for SFL):**
    * `electronics-13-03959-v2.pdf` (AWDP-FL): Adaptive Weight-based DP, layer-level processing, historical gradients.
    * `2401.02453v1.pdf` (Priority-Based Adaptive DP): Perturbing weights based on feature importance.
    * `2211.15893v1 (1).pdf` (Adap DP-FL): Adaptive gradient clipping and decreasing noise based on convergence.
* **Example SFL Implementation (Reference):**
    * [https://github.com/chandra2thapa/SplitFed-When-Federated-Learning-Meets-Split-Learning](https://github.com/chandra2thapa/SplitFed-When-Federated-Learning-Meets-Split-Learning)

**Note:** While some papers focus on standard Federated Learning (FL), the adaptive DP concepts presented should be adapted and applied within the SFL framework, specifically to the gradients/updates communicated between clients and the server.

---

## 4. Proposed File Structure

Organize the code repository using the following structure:

```plaintext
/sfl_adaptive_dp_repo
|
|-- /src
|   |-- __init__.py
|   |-- /models                # Neural network model definitions
|   |   |-- __init__.py
|   |   |-- base_model.py
|   |   |-- resnet.py
|   |   |-- vgg.py
|   |   |-- ... (other models)
|   |
|   |-- /datasets              # Data loading and partitioning logic
|   |   |-- __init__.py
|   |   |-- data_loader.py
|   |   |-- partition.py       # IID/Non-IID partitioning
|   |
|   |-- /sfl                   # Core SFL logic
|   |   |-- __init__.py
|   |   |-- client.py          # Client-side logic and model part
|   |   |-- server.py          # Server-side logic and model part
|   |   |-- utils.py           # SFL specific utilities
|   |
|   |-- /dp                    # Differential Privacy mechanisms
|   |   |-- __init__.py
|   |   |-- mechanisms.py      # DP noise addition, clipping
|   |   |-- adaptive.py        # Adaptive DP logic (clipping, noise)
|   |   |-- privacy_accountant.py # Privacy budget tracking
|   |
|   |-- /utils                 # General utilities (logging, config parsing)
|   |   |-- __init__.py
|   |   |-- logger.py
|   |   |-- config.py
|
|-- /experiments             # Experiment scripts and results
|   |-- train_sfl_dp.py        # Main training script
|   |-- run_experiments.sh     # Example script to run multiple experiments
|   |-- /results               # Directory to save logs/outputs
|
|-- requirements.txt         # Python package dependencies
|-- README.md                # Project description, setup, usage instructions
|-- .gitignore
```
