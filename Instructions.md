1. Introduction

This document outlines the requirements for a Python-based framework implementing Secure Split Federated Learning (SFL). The framework aims to simulate the SFL process, specifically the SFLV1 variant described by Thapa et al. (2022), incorporating privacy-enhancing noise mechanisms (Laplacian and Gaussian) as detailed in the same paper. A critical requirement is the manual implementation of these noise mechanisms and a corresponding manual privacy accountant (Moments Accountant) without relying on external libraries like Opacus or dp-accounting. This framework will serve as a foundation for research and experimentation in secure and private distributed learning, allowing for future extensions.

2. Goals

G1: Implement a functional simulation of the Split Federated Learning (SFLV1) architecture with distinct Client, Main Server, and Fed Server components.

G2: Integrate two specific privacy-enhancing noise mechanisms (Laplacian on activations, Gaussian with per-sample clipping on gradients) exactly as described by Thapa et al. (2022), using manual implementation.

G3: Implement a manual Moments Accountant to track the cumulative privacy budget (ε, δ) incurred by the gradient noise mechanism.

G4: Provide a configurable and modular framework allowing researchers to easily modify parameters, model architectures, splitting points, and noise settings for experiments.

G5: Ensure the codebase is well-structured, documented (especially the manual DP components), and follows professional software development practices.

G6: Enable basic evaluation of model performance (test accuracy) and privacy cost (ε, δ).

3. Target Audience

ML Researchers and Developers investigating Split Federated Learning, Differential Privacy, and secure distributed systems.

The primary user (you) who will build upon this foundation for further research.

4. Core Functionalities

4.1. SFLV1 Framework Implementation
* Requirement: Simulate the SFLV1 workflow (Ref: Thapa et al., 2022, Alg 1 & System Diagram).
* Components:
* SFLClient: Manages local data, client-side model (WC), performs local computations (forward to cut layer, backward from activation gradients), applies noise, and communicates with servers.
* MainServer: Manages server-side model (WS), receives noisy smashed data (Ak,t) and labels (Yk), performs server-side forward/backward passes, sends activation gradients (∇Ak,t) back to clients, aggregates server-side updates (∇WS_k).
* FedServer: Manages global client-side model (WC), receives noisy client-side updates (∇WCk,t or parameters), aggregates them, and distributes the updated WC.
* Communication Flow: Implement the round-based interaction: FedServer -> Clients (WC), Clients -> MainServer (noisy Ak,t, Yk), MainServer -> Clients (∇Ak,t), MainServer aggregates WS, Clients -> FedServer (noisy ∇WCk,t), FedServer aggregates WC.

4.2. Noise Mechanism 1: Laplacian Noise on Smashed Data
* Requirement: Add Laplacian noise to client activations before sending to the Main Server (Ref: Thapa et al., 2022, Eq 5 & PixelDP concept).
* Implementation: Manual. Use torch.distributions.laplace or manual generation from Uniform.
* Location: SFLClient, after computing Ak,t.
* Sensitivity (Δp,q): Assume a fixed, configurable value (e.g., config.laplacian_sensitivity).
* Parameter: Configurable privacy budget epsilon_prime (config.epsilon_prime).
* Formula: Noise scale = sensitivity / epsilon_prime.

4.3. Noise Mechanism 2: Gaussian Noise on Client Gradients
* Requirement: Apply differential privacy (clipping + Gaussian noise) to client-side gradients before sending to Fed Server (Ref: Thapa et al., 2022, Eq 2 & 3).
* Implementation: Manual. NO Opacus.
* Location: SFLClient, after computing ∇WCk,t.
* Per-Sample Gradient Clipping:
* Method: Implement using a micro-batching loop within the client's backward computation. For each sample in the batch, compute its individual gradient w.r.t WC, calculate its L2 norm, and clip it based on config.clip_norm.
* Documentation: Clearly document this micro-batching process in the code.
* Aggregation: Sum the clipped per-sample gradients.
* Noise Addition: Add Gaussian noise N(0, (noise_multiplier * clip_norm)^2 * I) to the summed clipped gradients.
* Parameters: Configurable clip_norm, noise_multiplier.

4.4. Privacy Accounting: Manual Moments Accountant
* Requirement: Track the cumulative privacy cost (ε, δ) incurred only by Noise Mechanism 2 over the training duration.
* Implementation: Manual. NO dp-accounting library.
* Method: Implement the Moments Accountant logic (Ref: Abadi et al., 2016, as used by Thapa et al.).
* Location: Likely a separate utility class ManualPrivacyAccountant.
* Functionality:
* __init__(...): Initialize necessary parameters (e.g., list of moment orders alphas).
* _compute_log_moment(...): Internal function to calculate the log moment for the Gaussian mechanism for a given alpha, noise multiplier, and sampling rate.
* step(noise_multiplier, sampling_rate, num_steps): Update the accumulated log moments for all alphas based on the steps taken.
* get_privacy_spent(delta): Compute the minimum ε for the target delta based on the accumulated moments.
* Integration: The training loop should call the accountant's step method appropriately after each client update involving Mechanism 2 noise.

4.5. Model Splitting
* Requirement: Allow splitting a given PyTorch model (e.g., CNN) into WC and WS at a configurable layer index (config.cut_layer).
* Implementation: Provide utility functions or a wrapper class for model splitting.

4.6. Aggregation Algorithms
* Requirement: Implement basic Federated Averaging (FedAvg) for both the Main Server (aggregating ∇WS_k or WS updates) and the Fed Server (aggregating noisy ∇WCk,t or WC updates). Structure should allow plugging in other algorithms later.

4.7. Configuration Management
* Requirement: All key parameters should be configurable externally.
* Method: Use a configuration file (e.g., YAML) or command-line arguments (argparse).
* Parameters: Learning rate, optimizer choice, batch size, number of clients, dataset partitioning type (start IID), total communication rounds, model architecture, cut_layer, laplacian_sensitivity, epsilon_prime, clip_norm, noise_multiplier, target delta (for accountant output), random seeds, etc.

4.8. Evaluation
* Requirement: Evaluate the trained global model (combined WC and WS) on a standard test dataset.
* Metrics: Report final test accuracy and the final computed privacy budget (ε, δ) from the manual accountant.

5. Non-Functional Requirements

NFR1: Code Modularity: Structure the code into logical modules/classes (see Section 7).

NFR2: Readability & Documentation: Write clean, readable Python code. Add comments, especially explaining the manual DP implementations (micro-batching, accountant logic).

NFR3: Extensibility: Design components (like aggregation, noise addition) to be potentially replaceable or extensible in the future.

NFR4: Correctness: Ensure the SFL flow and DP mechanisms precisely match the specified requirements and paper descriptions.

6. Design & Architecture

Stack: Python 3.x, PyTorch.

High-Level Structure: Client-Server simulation. Classes for SFLClient, MainServer, FedServer, SplitModel, ManualNoiseUtils, ManualPrivacyAccountant, DataLoader, ConfigManager.

7. Proposed File Structure

secure-sfl-framework/
│
├── configs/                 # Configuration files (e.g., default.yaml)
│
├── data/                    # Placeholder for downloaded datasets
│
├── experiments/             # Main training and evaluation scripts
│   └── train_secure_sfl.py
│
├── src/                     # Source code
│   ├── __init__.py
│   ├── datasets/            # Data loading and partitioning
│   │   ├── __init__.py
│   │   └── data_loader.py
│   ├── dp/                  # Differential privacy components
│   │   ├── __init__.py
│   │   ├── noise_utils.py   # MANUAL noise functions
│   │   └── privacy_accountant.py # MANUAL accountant class
│   ├── models/              # Model definitions and splitting logic
│   │   ├── __init__.py
│   │   ├── simple_cnn.py    # Example model
│   │   └── split_utils.py   # Model splitting functions/class
│   ├── sfl/                 # SFL core components
│   │   ├── __init__.py
│   │   ├── client.py
│   │   ├── main_server.py
│   │   ├── fed_server.py
│   │   └── aggregation.py   # FedAvg implementation
│   └── utils/               # Utility functions
│       ├── __init__.py
│       └── config_parser.py # Argparse or config file loading
│
├── tests/                   # Unit tests (Optional but Recommended)
│
├── requirements.txt         # Python dependencies
├── README.md                # Project overview, setup, execution instructions
└── .gitignore


8. Experiments & Configuration

Execution: The main script (experiments/train_secure_sfl.py) should take configuration parameters (via file or CLI).

Workflow: Initialize system -> Load/Partition Data -> Run SFL rounds (including noise and accounting) -> Evaluate final model -> Report accuracy and privacy loss.

Configurability: Enable easy modification of parameters listed in 4.7 to study their impact on accuracy and privacy.

Logging: Implement basic logging of training progress (round, loss) and final results.

9. README Generation

The README.md file must include:

Overview: Brief description of the project and its purpose.

Features: List key implemented features (SFLV1, manual noise types, manual accountant).

File Structure: Explanation of the directory layout.

Setup: Instructions for setting up the environment (Python version, pip install -r requirements.txt).

Configuration: Explanation of how to configure experiments (e.g., modifying configs/default.yaml or using CLI arguments), including descriptions of key parameters.

Running Experiments: Clear command-line examples for running the main training script (e.g., python experiments/train_secure_sfl.py --config configs/mnist_default.yaml).

Expected Output: Description of the output logs and final results (accuracy, privacy budget).

10. Out of Scope (v1.0)

Advanced Non-IID data partitioning strategies.

Other SFL variants (e.g., SFLV2).

Advanced aggregation algorithms beyond FedAvg.

Dynamic sensitivity calculation for Laplacian noise.

Support for other DP mechanisms or noise types.

Robustness against specific attacks (focus is on privacy implementation).

Graphical User Interface (GUI).

Distributed deployment (simulation only).