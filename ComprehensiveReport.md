# Comprehensive Report: Secure Split Federated Learning Framework

## 1. Introduction

This repository implements a Secure Split Federated Learning (SFL) framework with differential privacy guarantees. The project serves as a research platform for exploring privacy-preserving techniques in distributed machine learning, specifically focusing on the SFLV1 variant as described by Thapa et al. (2022). The framework includes two key privacy-enhancing mechanisms: Laplacian noise on activations and Gaussian noise on gradients with per-sample clipping, as well as manual privacy accounting to track privacy budgets.

The repository also implements an adaptive differential privacy approach based on the "Adap DP-FL" paper by Fu et al. (2022), which dynamically adjusts clipping thresholds and noise scales based on model convergence.

## 2. Background and Theoretical Foundations

### 2.1 Split Federated Learning (SFLV1)

Split Federated Learning combines aspects of Split Learning and Federated Learning:

- **Split Learning**: Divides a neural network into client and server portions, with clients computing up to a cut layer and the server completing the forward pass.
- **Federated Learning**: Allows multiple clients to train a shared model without centralizing data.
- **SFLV1**: A variant where clients process their data up to a cut layer, send "smashed data" (activations) to a main server that completes the forward/backward computation, and then clients finish the backward pass. A federation server aggregates client model updates.

### 2.2 Differential Privacy in Deep Learning

The framework implements two noise mechanisms:

1. **Laplacian Noise (Mechanism 1)**: Applied to activations before they're sent to the main server, providing pixel-level differential privacy.
2. **Gaussian Noise (Mechanism 2)**: Applied to gradients with per-sample clipping before sending to the federation server.

### 2.3 Privacy Accounting

The Moments Accountant tracks the privacy loss over multiple training rounds, converting Renyi Differential Privacy (RDP) to (ε, δ)-differential privacy guarantees.

### 2.4 Adaptive Differential Privacy

The adaptive approach dynamically adjusts:
- Client-specific clipping thresholds based on gradient statistics
- Global noise scale based on validation loss trends

## 3. Architecture Overview

The framework follows a three-component architecture:

1. **Clients**: Hold local data, compute forward passes to the cut layer, receive activation gradients from the server, complete backward passes, and send model updates.
2. **Main Server**: Receives activations and labels, completes forward passes, computes gradients, and sends activation gradients back to clients.
3. **Federation Server**: Aggregates client model updates and distributes the global model.

![Architecture Diagram (Conceptual)]
```
                   ┌─────────────────┐
                   │                 │
                   │ Federation      │
                   │ Server          │
                   │                 │
                   └─────┬───────────┘
                         │
                 Global Model Distribution
                         │
                   ┌─────▼───────────┐
             ┌─────┤                 ├─────┐
             │     │ Clients         │     │
             │     │                 │     │
             │     └─────┬───────────┘     │
             │           │                 │
   Noisy Activations   Noisy         Activation
   + Labels            Gradients     Gradients
             │           │                 │
             │     ┌─────▼───────────┐     │
             └────►│                 │◄────┘
                   │ Main Server     │
                   │                 │
                   └─────────────────┘
```

## 4. Detailed Component Analysis

### 4.1 Dataset Management (`src/datasets`)

The `data_loader.py` module handles:
- Loading standard datasets (currently MNIST)
- Partitioning data among clients (IID partitioning)
- Creating validation sets for adaptive privacy mechanisms

### 4.2 Models (`src/models`)

The repository includes:
- `simple_cnn.py`: A basic CNN architecture for image classification
- `simple_dnn.py`: A dense neural network for comparison
- `split_utils.py`: Utilities for splitting models at specified layers

The model splitting functionality is crucial for SFL, as it divides a neural network into client and server portions based on a configurable cut layer.

### 4.3 Differential Privacy Components (`src/dp`)

#### 4.3.1 Noise Utilities (`noise_utils.py`)

Implements manual noise mechanisms:
- `add_laplacian_noise()`: Adds Laplacian noise to activations with scale determined by sensitivity/epsilon_prime
- `add_gaussian_noise()`: Adds Gaussian noise scaled by clip_norm and noise_multiplier to gradients
- `clip_gradients()`: Helper for gradient clipping

#### 4.3.2 Privacy Accountant (`privacy_accountant.py`)

The `ManualPrivacyAccountant` class:
- Tracks privacy loss over training rounds
- Computes RDP for the sampled Gaussian mechanism
- Converts RDP to (ε, δ)-DP guarantees
- Supports adaptive noise scales

Key methods:
- `_compute_rdp_epsilon_step()`: Computes RDP for a single step
- `step()`: Updates accumulated privacy cost
- `get_privacy_spent()`: Computes final (ε, δ) guarantee

### 4.4 Secure Split Federated Learning Components (`src/sfl`)

#### 4.4.1 Client (`client.py`)

The `SFLClient` class handles:
- Local data processing
- Forward pass to the cut layer (with optional Laplacian noise)
- Backward pass from activation gradients
- Per-sample gradient clipping and Gaussian noise addition
- Adaptive clipping threshold calculation

#### 4.4.2 Main Server (`main_server.py`)

The `MainServer` class manages:
- Receiving client activations and labels
- Server-side forward/backward passes
- Computing activation gradients
- Aggregating server-side model updates

#### 4.4.3 Federation Server (`fed_server.py`)

The `FedServer` class:
- Manages the global client-side model
- Aggregates client model updates
- Distributes updated models
- For adaptive DP, manages validation loss tracking and noise scale adjustment

#### 4.4.4 Aggregation (`aggregation.py`)

Implements FedAvg (Federated Averaging) for aggregating model updates from multiple clients.

### 4.5 Configuration and Utilities (`src/utils`)

The `config_parser.py` loads experiment configurations from YAML files, making the framework highly customizable.

## 5. Training and Workflow (`experiments`)

The `train_secure_sfl.py` script orchestrates the entire SFL training process:

1. Load configuration and datasets
2. Split models and initialize components (clients, servers)
3. For each communication round:
   - Distribute current model parameters
   - Clients compute forward passes and send activations
   - Main server computes activation gradients
   - Clients perform backward passes with noisy gradients
   - Federation server aggregates updates
   - For adaptive DP, update noise scale based on validation loss
4. Evaluate final model and report results

### 5.1 Fixed vs. Adaptive Differential Privacy

The repository supports two DP approaches:

**Fixed DP**:
- Constant clipping threshold across rounds
- Constant noise scale throughout training

**Adaptive DP**:
- Client-specific clipping thresholds adjusted based on gradient statistics
- Global noise scale decreased as model converges

## 6. Configuration System

The framework uses YAML configuration files, allowing researchers to easily set:

- Basic parameters (dataset, model, clients, rounds)
- Optimization parameters (learning rate, optimizer, batch size)
- DP parameters (sensitivities, noise multipliers, target delta)
- Adaptive parameters (clipping factor, noise decay factor, patience)

This configuration-driven approach facilitates reproducible experiments and parameter exploration.

## 7. Privacy-Utility Tradeoff Analysis

The framework enables researchers to explore the fundamental privacy-utility tradeoff in differentially private learning:

- Higher noise levels increase privacy (lower ε) but may decrease model accuracy
- Adaptive mechanisms aim to improve this tradeoff by reducing noise as the model converges
- Different model architectures and cut layer positions affect the privacy-utility balance

## 8. Extensibility and Future Work

The modular design allows for future extensions:

- Support for additional datasets and model architectures
- Implementation of non-IID data partitioning
- Integration of other SFL variants (SFLV2, etc.)
- Advanced aggregation algorithms beyond FedAvg
- Robustness against specific attacks

## 9. Conclusion

This SFL framework provides a foundation for researching privacy-preserving distributed learning techniques. By implementing both fixed and adaptive differential privacy approaches manually (without relying on external DP libraries), it offers transparency and control over privacy mechanisms while serving as an educational resource for understanding these concepts.

The detailed privacy accounting and configurable nature of the framework make it suitable for exploring the complex tradeoffs between model utility, privacy guarantees, and communication efficiency in secure distributed learning scenarios. 