import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset, Dataset
import numpy as np
import os
import sys
from .partition import partition_data_dirichlet

class BCWDataset(Dataset):
    """
    PyTorch Dataset for the Breast Cancer Wisconsin dataset from sklearn.
    """
    def __init__(self, features, labels, transform=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
            
        return x, y

def get_bcw_dataloaders(config):
    """
    Loads the Breast Cancer Wisconsin dataset from sklearn and creates DataLoaders.
    """
    try:
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("scikit-learn is not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
    
    batch_size = config['batch_size']
    
    # Load the breast cancer dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split the data into training (80%) and test (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=config['seed'])
    
    # Create PyTorch datasets
    train_dataset = BCWDataset(X_train, y_train)
    test_dataset = BCWDataset(X_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, train_dataset, test_dataset

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

def get_dataloaders(config):
    """Returns train and test loaders based on the dataset specified in config."""
    dataset_name = config.get('dataset', 'mnist').lower()
    
    if dataset_name == 'mnist':
        return get_mnist_dataloaders(config)
    elif dataset_name == 'bcw':
        return get_bcw_dataloaders(config)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}") 