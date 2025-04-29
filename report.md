# Implementation Report: SFL with Adaptive DP

This report summarizes the implementation of the Split Federated Learning (SFL) framework with Differential Privacy (DP) capabilities, based on the requirements outlined in `Instructions.md`.

## 1. Project Structure

The code adheres to the proposed file structure:

```plaintext
/sfl_adaptive_dp_repo
|-- /src
|   |-- /models (base_model.py, resnet.py, vgg.py, simple_cnn.py)
|   |-- /datasets (data_loader.py, partition.py)
|   |-- /sfl (client.py, server.py)
|   |-- /dp (mechanisms.py, adaptive.py, privacy_accountant.py)
|   |-- /utils (logger.py, config.py)
|-- /experiments
|   |-- train_sfl_dp.py
|   |-- /results (.gitkeep)
|   |-- run_experiments.sh (empty)
|-- requirements.txt
|-- README.md
|-- .gitignore
```

## 2. Core Functionalities Implemented

### 2.1. Split Federated Learning Framework (`src/sfl`)

*   **Architecture:** Implements SFL v2 (sequential client-server interaction per batch).
*   **Client (`client.py`):**
    *   Holds the client-side model part.
    *   Performs forward pass up to the cut layer.
    *   Manages local optimizer (SGD, Adam).
    *   Simulates sending smashed data (activations) to the server.
    *   Receives gradients from the server and performs backward pass and optimizer step.
    *   Supports multiple local epochs (`--local_epochs`).
*   **Server (`server.py`):**
    *   Holds the server-side model part.
    *   Receives smashed data from clients.
    *   Performs forward pass on the server model.
    *   Calculates loss and performs backward pass.
    *   Sends gradients back to the corresponding clients.
    *   Updates its own model part using its optimizer (SGD, Adam).
    *   Includes an `evaluate` method for testing the combined model on a global test set.
*   **Model Splitting (`src/models/base_model.py`):**
    *   A `SplitModelBase` class facilitates splitting `nn.Module` based models.
    *   A `get_model` factory function creates and splits specified models (`ResNet18`, `VGG11`, `SimpleCNN`) based on a layer name (`--split_layer`). Models inherit from `SplitModelBase` and define named layers/blocks for splitting.

### 2.2. Differential Privacy Mechanisms (`src/dp`)

*   **Vanilla DP (`mechanisms.py`):**
    *   Utilizes the `Opacus` library (`PrivacyEngine`) to implement standard (Gaussian) DP.
    *   Attaches DP to the *client-side* model part and optimizer (`attach_dp_mechanism` function).
    *   Configurable via `--dp_mode vanilla`, `--noise_multiplier`, `--max_grad_norm`.
    *   Uses Poisson sampling within Opacus for potentially better privacy amplification.
*   **Adaptive DP (`adaptive.py`, `mechanisms.py`):**
    *   Placeholders are implemented for `--dp_mode adaptive_clipping`, `adaptive_noise`, `awdp`.
    *   Currently, these modes functionally fallback to the vanilla DP setup during `attach_dp_mechanism`.
    *   The `adaptive.py` file contains commented-out conceptual logic and placeholder functions (`update_clipping_norm_adaptive`, `update_noise_multiplier_adaptive`, etc.) indicating where the specific adaptive algorithms (based on referenced papers) would be integrated.
    *   **Note:** Full implementation of adaptive mechanisms requires significant custom code interacting with the training loop and potentially Opacus internals (or manual DP implementation).
*   **Privacy Accounting (`mechanisms.py`, `privacy_accountant.py`):
    *   Uses Opacus' RDP accountant (`RDPAccountant`) via the `PrivacyEngine`.
    *   The `get_privacy_spent` function retrieves the total epsilon consumed for a given delta (`--target_delta`).
    *   The `privacy_accountant.py` file is largely a placeholder, as the main accounting is handled by Opacus within `mechanisms.py`.

### 2.3. Experiment Execution and Configuration

*   **Main Script (`experiments/train_sfl_dp.py`):**
    *   Parses command-line arguments using `argparse` (`src/utils/config.py`).
    *   Sets up random seeds and device (CPU/GPU).
    *   Loads datasets (MNIST, FashionMNIST, CIFAR10) and partitions data (IID/Non-IID via Dirichlet) using `src/datasets` modules.
    *   Initializes the SFL server and clients, deep-copying model parts.
    *   Attaches DP mechanisms via `attach_dp_mechanism` if specified.
    *   Runs the SFL training loop round by round:
        *   Selects clients for the round (`--clients_per_round`).
        *   Simulates batch-wise interaction: clients compute smashed data, server processes, server sends gradients, clients update.
        *   Handles local epochs on clients.
        *   Calls placeholder adaptive DP update functions (currently inactive).
        *   Evaluates the model on the test set after each round.
        *   Calculates and logs approximate total privacy budget consumed.
        *   Logs results (loss, accuracy, epsilon) to console and a CSV file (`src/utils/logger.py`).
        *   Includes an early stopping condition based on `--target_epsilon`.
*   **Configuration (`src/utils/config.py`):**
    *   Provides extensive command-line arguments to control all aspects of the experiment.
*   **Logging (`src/utils/logger.py`):**
    *   Logs to console and a structured CSV file in `experiments/results/`.

### 2.4. Evaluation and Correctness Check

*   **Evaluation:** Standard accuracy is calculated on the global test set by the server after each round (`server.evaluate`).
*   **Correctness:**
    *   Opacus handles the correct application of noise and clipping for the `vanilla` DP mode.
    *   Assertions are used in data partitioning (`partition.py`) to check index assignments.
    *   The simulation logic ensures gradients are routed back to the correct clients after server processing.
    *   **Note:** Correctness checks for *adaptive* DP mechanisms would need to be added alongside their implementation.

## 3. Key Libraries Used

*   **PyTorch:** Core ML framework.
*   **Opacus:** For Differential Privacy implementation (Gaussian mechanism, privacy accounting).
*   **NumPy:** For numerical operations, especially in data partitioning.
*   **Torchvision:** For standard datasets and transforms.
*   **Argparse:** For command-line argument parsing.
*   **Tqdm:** For progress bars (currently not integrated into the main loop but available).

## 4. Limitations and Future Work

*   **Adaptive DP:** The core limitation is the placeholder status of adaptive DP mechanisms. Implementing algorithms like AWDP or Priority-Based DP requires further development.
*   **Communication Simulation:** The communication is simulated via direct function calls and object passing. Real-world network latency and bandwidth are not modeled.
*   **SFL Simulation:** The training loop simulates sequential batch processing across selected clients for simplicity. True asynchronous or parallel client execution is not implemented.
*   **Opacus in SFL:** Applying Opacus (designed for centralized or standard FL) directly to the *client part* in SFL is a simulation choice. The privacy implications and sensitivity analysis might differ from applying DP to the full model or gradients aggregated at the server.
*   **Evaluation Model:** Evaluation uses the model state from Client 0 combined with the server. Averaging client-side models before evaluation could be an alternative.

## 5. Example Run (Trial)

To demonstrate the core SFL functionality (without DP) on MNIST:

```bash
cd sfl_adaptive_dp_repo
python experiments/train_sfl_dp.py \
    --dataset MNIST \
    --num_clients 5 \
    --model SimpleCNN \
    --split_layer pool1 \
    --epochs 3 \
    --dp_mode none \
    --batch_size 128 \
    --lr 0.01 \
    --log_filename trial_run_log.csv
```

This command trains a SimpleCNN split at `pool1` for 3 rounds using 5 clients on the MNIST dataset without DP. Results (accuracy, loss) will be printed to the console and logged in `experiments/results/trial_run_log.csv`.

To demonstrate Vanilla DP:

```bash
cd sfl_adaptive_dp_repo
python experiments/train_sfl_dp.py \
    --dataset MNIST \
    --num_clients 5 \
    --model SimpleCNN \
    --split_layer pool1 \
    --epochs 3 \
    --dp_mode vanilla \
    --target_epsilon 1.0 \
    --target_delta 1e-5 \
    --noise_multiplier 1.1 \
    --max_grad_norm 1.0 \
    --batch_size 128 \
    --lr 0.01 \
    --log_filename trial_run_dp_log.csv
```
This command runs a similar setup but enables vanilla DP with specified privacy parameters. The log will include consumed epsilon values. 