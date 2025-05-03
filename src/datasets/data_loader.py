import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
from .partition import iid_partition, non_iid_partition

def get_datasets(dataset_name, data_root):
    """Loads the specified dataset."""
    transform = None
    if dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    elif dataset_name == 'FashionMNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=transform)
    elif dataset_name == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return train_dataset, test_dataset

def get_data_loaders(args, train_dataset, test_dataset):
    """Partitions the training data and creates dataloaders."""
    if args.data_distribution == 'iid':
        client_datasets = iid_partition(train_dataset, args.num_clients)
    elif args.data_distribution == 'non-iid':
        client_datasets = non_iid_partition(train_dataset, args.num_clients, args.non_iid_alpha)
    else:
        raise ValueError(f"Unsupported data distribution: {args.data_distribution}")

    client_loaders = [
        DataLoader(subset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        for subset in client_datasets
    ]

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)

    return client_loaders, test_loader 