import numpy as np
from torch.utils.data import Subset
import torch

def iid_partition(dataset, num_clients):
    """Partitions the dataset into IID subsets for each client."""
    num_items_per_client = len(dataset) // num_clients
    client_indices = {}
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    for i in range(num_clients):
        start_idx = i * num_items_per_client
        end_idx = (i + 1) * num_items_per_client if i != num_clients - 1 else len(dataset)
        client_indices[i] = indices[start_idx:end_idx]
    return [Subset(dataset, client_indices[i]) for i in range(num_clients)]

def non_iid_partition(dataset, num_clients, alpha=0.5):
    """Partitions the dataset into Non-IID subsets using Dirichlet distribution."""
    try:
        num_classes = len(dataset.classes)
        labels = np.array(dataset.targets) # Use targets attribute
    except AttributeError:
        # Handle datasets that might not have .classes or .targets directly (e.g., TensorDataset)
        # This implementation assumes labels are the second element in each data point tuple
        # You might need to adjust this based on your specific dataset structure
        if isinstance(dataset, torch.utils.data.TensorDataset):
             # Assuming targets are the second tensor in the TensorDataset
            labels = dataset.tensors[1].cpu().numpy()
            num_classes = len(np.unique(labels))
            if labels is None:
                 raise ValueError("Could not extract labels for non-IID partitioning from TensorDataset.")
        else:
             raise ValueError("Dataset type not supported for automatic non-IID partitioning without explicit labels.")


    num_items = len(dataset)
    client_indices = {i: [] for i in range(num_clients)}
    indices_per_class = {c: np.where(labels == c)[0] for c in range(num_classes)}

    # Ensure all classes have samples
    for c in range(num_classes):
        if len(indices_per_class[c]) == 0:
            print(f"Warning: Class {c} has no samples in the dataset.")
            # Optionally handle this case, e.g., by skipping or raising an error
            # For now, we proceed, but clients might not get data for this class

    # Distribute indices using Dirichlet distribution
    for c in range(num_classes):
        if len(indices_per_class[c]) > 0:
            class_indices = indices_per_class[c]
            np.random.shuffle(class_indices)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))

            # Ensure proportions sum roughly to the number of class indices
            proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)
            proportions = np.diff(np.concatenate(([0], proportions)))
             # Correct potential rounding issues, ensure total count matches
            proportions[-1] = len(class_indices) - np.sum(proportions[:-1])
            assert np.sum(proportions) == len(class_indices), f"Proportions sum error for class {c}"


            start_idx = 0
            for client_id in range(num_clients):
                num_samples_for_client = proportions[client_id]
                client_indices[client_id].extend(class_indices[start_idx : start_idx + num_samples_for_client])
                start_idx += num_samples_for_client
            assert start_idx == len(class_indices), f"Index assignment error for class {c}"


    # Shuffle indices within each client's list
    for client_id in range(num_clients):
        np.random.shuffle(client_indices[client_id])

    return [Subset(dataset, client_indices[i]) for i in range(num_clients)] 