import argparse
import torch
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Split Federated Learning with Adaptive DP')

    # General arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cpu, cuda)')
    parser.add_argument('--log_dir', type=str, default='experiments/results', help='Directory to save logs')
    parser.add_argument('--log_filename', type=str, default='experiment_log.csv', help='Filename for logging results')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, required=True, choices=['MNIST', 'FashionMNIST', 'CIFAR10'], help='Dataset to use')
    parser.add_argument('--data_root', type=str, default='./data', help='Root directory for dataset storage')
    parser.add_argument('--data_distribution', type=str, default='iid', choices=['iid', 'non-iid'], help='Data distribution across clients (iid or non-iid)')
    parser.add_argument('--non_iid_alpha', type=float, default=0.5, help='Concentration parameter for Dirichlet distribution for non-iid data')

    # SFL arguments
    parser.add_argument('--num_clients', type=int, required=True, help='Number of clients')
    parser.add_argument('--clients_per_round', type=int, default=None, help='Number of clients participating in each round (default: all)')
    parser.add_argument('--model', type=str, required=True, help='Model architecture (e.g., SimpleCNN, ResNet18, VGG11)')
    parser.add_argument('--split_layer', type=str, required=True, help='Name of the layer where the model is split')
    parser.add_argument('--epochs', type=int, required=True, help='Number of communication rounds (server epochs)')
    parser.add_argument('--local_epochs', type=int, default=1, help='Number of local training epochs on client')

    # Training arguments
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'], help='Optimizer to use')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')

    # DP arguments
    parser.add_argument("--dp_mode", type=str, default="none", choices=["none", "vanilla", "adaptive_clipping", "adaptive_noise", "awdp", "adaptive_trusted_client"], help="DP mode")
    parser.add_argument("--target_epsilon", type=float, default=1.0, help="Target privacy budget epsilon (for adaptive modes and privacy accountant)")
    parser.add_argument("--target_delta", type=float, default=1e-5, help="Target privacy budget delta")
    parser.add_argument("--noise_multiplier", type=float, default=1.0, help="Noise multiplier (sigma) for Gaussian mechanism (used in vanilla and adaptive_noise)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Clipping norm (C) (used in vanilla and initial value for adaptive_clipping/awdp)")

    # Adaptive DP arguments (specific)
    parser.add_argument("--adaptive_clipping_quantile", type=float, default=0.5, help="Quantile for adaptive clipping (e.g., 0.5 for median)")
    parser.add_argument("--adaptive_noise_decay", type=float, default=0.99, help="Decay factor for adaptive noise schedule")
    parser.add_argument("--awdp_layerwise", action="store_true", help="Enable layer-wise adaptivity for AWDP")

    # Adaptive Trusted Client DP arguments
    parser.add_argument("--trusted_client_id", type=int, default=0, help="ID of the trusted client responsible for DP parameter adaptation")
    parser.add_argument("--feedback_metric", type=str, default="grad_norm", choices=["grad_norm"], help="Metric used by clients for feedback (currently only grad_norm supported)")
    parser.add_argument("--adaptive_step_size", type=float, default=0.05, help="Step size for adjusting sigma/C in adaptive_trusted_client mode")
    parser.add_argument("--min_sigma", type=float, default=0.1, help="Minimum noise multiplier (sigma)")
    parser.add_argument("--max_sigma", type=float, default=5.0, help="Maximum noise multiplier (sigma)")
    parser.add_argument("--min_C", type=float, default=0.1, help="Minimum clipping norm (C)")
    parser.add_argument("--max_C", type=float, default=10.0, help="Maximum clipping norm (C)")

    args = parser.parse_args()

    # Create log directory if it doesn't exist
    os.makedirs(args.log_dir, exist_ok=True)

    if args.clients_per_round is None:
        args.clients_per_round = args.num_clients # Default to using all clients if not specified

    return args 