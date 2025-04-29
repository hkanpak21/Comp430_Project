# Split Federated Learning with Adaptive Differential Privacy

This repository provides a Python implementation for experimenting with Split Federated Learning (SFL) combined with Differential Privacy (DP), including vanilla and placeholder adaptive mechanisms.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd sfl_adaptive_dp_repo
    ```
2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The main script for running experiments is `experiments/train_sfl_dp.py`.

**Example Commands:**

*   **Train SFL without DP on MNIST:**
    ```bash
    python experiments/train_sfl_dp.py --dataset MNIST --num_clients 10 --dp_mode none --epochs 5 --model SimpleCNN --split_layer pool1 --batch_size 128 --lr 0.01
    ```

*   **Train SFL with Vanilla DP on FashionMNIST:**
    ```bash
    python experiments/train_sfl_dp.py --dataset FashionMNIST --num_clients 5 --dp_mode vanilla --target_epsilon 1.0 --target_delta 1e-5 --noise_multiplier 1.1 --max_grad_norm 1.0 --epochs 10 --model SimpleCNN --split_layer pool1 --batch_size 64
    ```

*   **Train SFL (Placeholder Adaptive DP) on CIFAR-10:**
    ```bash
    # Note: Adaptive mechanisms are currently placeholders using Vanilla DP settings.
    python experiments/train_sfl_dp.py --dataset CIFAR10 --num_clients 20 --dp_mode adaptive_clipping --target_epsilon 2.0 --target_delta 1e-5 --noise_multiplier 1.0 --max_grad_norm 1.2 --epochs 20 --model ResNet18 --split_layer layer1 --batch_size 256 --lr 0.05
    ```

**Command-line Arguments:**

Run `python experiments/train_sfl_dp.py -h` to see all available arguments for configuration, including:
*   Dataset (`--dataset`, `--data_root`, `--data_distribution`, `--non_iid_alpha`)
*   SFL setup (`--num_clients`, `--clients_per_round`, `--model`, `--split_layer`, `--epochs`, `--local_epochs`)
*   Training (`--optimizer`, `--lr`, `--batch_size`)
*   DP (`--dp_mode`, `--target_epsilon`, `--target_delta`, `--noise_multiplier`, `--max_grad_norm`)
*   Adaptive DP placeholders (`--adaptive_clipping_quantile`, etc.)
*   General (`--seed`, `--device`, `--log_dir`, `--log_filename`)

## Project Structure

Refer to the `Instructions.md` document for the detailed file structure and component descriptions.

## Notes

*   The adaptive DP mechanisms (`adaptive_clipping`, `adaptive_noise`, `awdp`) are currently implemented as placeholders using the vanilla DP setup via Opacus. Full implementation requires significant custom logic based on the referenced academic papers.
*   Privacy accounting uses Opacus' RDP accountant.
*   The simulation assumes perfect communication between clients and the server.
*   Evaluation uses the global test set, processed by the server using one client's final model part (client 0 by default).
