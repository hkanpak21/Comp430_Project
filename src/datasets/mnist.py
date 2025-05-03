import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import logging

def get_mnist_datasets(data_path='./data'):
    """Downloads and loads MNIST train and test datasets."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def split_mnist_data(train_dataset, num_clients, split_mode='iid', non_iid_alpha=0.1, seed=42):
    """Splits the MNIST training dataset among clients.

    Args:
        train_dataset: The full MNIST training dataset.
        num_clients: The number of clients.
        split_mode: 'iid' or 'non-iid'.
        non_iid_alpha: Dirichlet distribution alpha parameter for non-iid split.
                       Smaller alpha -> more non-iid.
        seed: Random seed for reproducibility.

    Returns:
        A list of Subset datasets, one for each client.
        A dictionary mapping client_id to class distribution counts.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    num_samples = len(train_dataset)
    indices = list(range(num_samples))
    client_data_indices = [[] for _ in range(num_clients)]
    labels = np.array(train_dataset.targets)
    num_classes = len(np.unique(labels))
    client_class_distribution = {i: {c: 0 for c in range(num_classes)} for i in range(num_clients)}

    if split_mode == 'iid':
        logging.info("Using IID data split.")
        np.random.shuffle(indices)
        samples_per_client = num_samples // num_clients
        if num_samples % num_clients != 0:
            logging.warning(f"Discarding {num_samples % num_clients} samples due to uneven division.")

        for client_id in range(num_clients):
            start_idx = client_id * samples_per_client
            end_idx = start_idx + samples_per_client
            client_indices = indices[start_idx:end_idx]
            client_data_indices[client_id] = client_indices
            # Record class distribution
            for idx in client_indices:
                 label = labels[idx]
                 client_class_distribution[client_id][label] += 1

    elif split_mode == 'non-iid':
        logging.info(f"Using Non-IID data split with alpha={non_iid_alpha}.")
        if non_iid_alpha <= 0:
            raise ValueError("non_iid_alpha must be positive for Dirichlet distribution.")

        # Ensure indices are sorted by label for Dirichlet distribution application
        # Instead of sorting the whole dataset, get indices per class directly
        label_indices = [np.where(labels == i)[0] for i in range(num_classes)]
        class_proportions = np.random.dirichlet([non_iid_alpha] * num_clients, num_classes)

        # Make sure each client gets *some* data, even if proportions are tiny
        # Assign minimum samples per class per client if possible
        min_samples_per_class_client = 1

        assigned_indices = set()

        for client_id in range(num_clients):
            client_indices_for_this_client = [] # Rename to avoid confusion
            for class_id in range(num_classes):
                # Find indices for this class that haven't been assigned yet
                available_indices_for_class = np.setdiff1d(label_indices[class_id], list(assigned_indices))
                target_num_samples = int(class_proportions[class_id, client_id] * len(label_indices[class_id]))

                 # Give at least min_samples if possible and proportion is non-zero and samples available
                if target_num_samples == 0 and class_proportions[class_id, client_id] > 0 and len(available_indices_for_class) > 0:
                     target_num_samples = min_samples_per_class_client

                num_to_assign = min(target_num_samples, len(available_indices_for_class))

                if num_to_assign > 0:
                    chosen_indices = np.random.choice(available_indices_for_class, num_to_assign, replace=False)
                    client_indices_for_this_client.extend(chosen_indices)
                    assigned_indices.update(chosen_indices)
                    client_class_distribution[client_id][class_id] += num_to_assign

            client_data_indices[client_id] = client_indices_for_this_client # Assign collected indices
            if not client_indices_for_this_client:
                 logging.warning(f"Client {client_id} received no data points during initial non-iid distribution (alpha={non_iid_alpha}). Check alpha or data size.")

        # Handle remaining unassigned indices (due to rounding or minimums) - distribute randomly among clients
        unassigned_indices = np.setdiff1d(indices, list(assigned_indices)).tolist()
        np.random.shuffle(unassigned_indices)
        logging.info(f"Distributing {len(unassigned_indices)} remaining samples randomly.")

        client_idx_turn = 0
        for idx in unassigned_indices:
             target_client = client_idx_turn % num_clients
             client_data_indices[target_client].append(idx)
             label = labels[idx]
             client_class_distribution[target_client][label] += 1
             client_idx_turn += 1

    else:
        raise ValueError(f"Unknown split_mode: {split_mode}. Choose 'iid' or 'non-iid'.")

    client_datasets = [Subset(train_dataset, indices) for indices in client_data_indices]

    # Log distribution summary
    logging.info("Client data distribution summary:")
    for cid, counts in client_class_distribution.items():
        total_samples = sum(counts.values())
        dist_str = ", ".join([f"{c}: {n}" for c, n in counts.items()]) # Show all counts
        logging.info(f" Client {cid}: {total_samples} samples -> ({dist_str})")
        if total_samples == 0:
             logging.warning(f"Client {cid} has 0 samples after distribution!")


    return client_datasets, client_class_distribution

def get_client_dataloader(dataset, batch_size):
    """Creates a DataLoader for a client's dataset subset."""
    if not dataset or len(dataset) == 0: # Check if dataset is empty
         logging.warning("Attempting to create DataLoader for an empty dataset.")
         return None # Handle cases where a client gets no data
    # Drop last batch if it's smaller than batch_size and might cause issues (e.g., BatchNorm)
    drop_last = len(dataset) % batch_size == 1
    if drop_last:
         logging.warning(f"Dropping last batch for client loader due to size 1.")

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)

def get_test_dataloader(test_dataset, batch_size):
    """Creates a DataLoader for the test dataset."""
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 