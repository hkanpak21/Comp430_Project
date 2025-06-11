import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import os
from .partition import partition_data_dirichlet

def get_mnist_dataloaders(config):
    """Downloads MNIST, applies transformations, and creates DataLoaders for train and test sets."""
    data_dir = config['data_dir']
    batch_size = config['batch_size']

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # MNIST specific normalization
    ])

    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Download and load the training data
    train_dataset = datasets.MNIST(root=data_dir,
                                   train=True,
                                   download=True,
                                   transform=transform)

    # Download and load the test data
    test_dataset = datasets.MNIST(root=data_dir,
                                  train=False,
                                  download=True,
                                  transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset, test_dataset

def partition_data_iid(dataset, num_clients):
    """Partitions a dataset into IID subsets for each client."""
    num_items_per_client = len(dataset) // num_clients
    client_datasets = {}
    all_indices = list(range(len(dataset)))
    np.random.shuffle(all_indices) # Shuffle indices for random distribution

    for i in range(num_clients):
        start_idx = i * num_items_per_client
        # Ensure the last client gets any remaining data points
        end_idx = (i + 1) * num_items_per_client if i != num_clients - 1 else len(dataset)
        client_indices = all_indices[start_idx:end_idx]
        client_datasets[i] = Subset(dataset, client_indices)

    return client_datasets

def get_client_data_loaders(config, train_dataset):
    """Partitions the training data and creates DataLoaders for each client."""
    num_clients = config['num_clients']
    batch_size = config['batch_size']
    partition_method = config.get('partition_method', 'iid')
    
    if partition_method == 'dirichlet':
        alpha = config.get('dirichlet_alpha', 1.0)
        print(f"Using Dirichlet partition with alpha={alpha}")
        client_datasets = partition_data_dirichlet(train_dataset, num_clients, alpha)
    else:
        # Default to IID partitioning
        print("Using IID partition")
        client_datasets = partition_data_iid(train_dataset, num_clients)
    
    client_loaders = {}
    for client_id, dataset in client_datasets.items():
        client_loaders[client_id] = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return client_loaders 