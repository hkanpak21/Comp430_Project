import numpy as np
import torch
from torch.utils.data import Subset

def partition_data_dirichlet(dataset, num_clients, alpha):
    """
    Partitions the data using a Dirichlet distribution to create non-IID data splits.
    
    Args:
        dataset: The dataset to partition (can be a regular dataset or a Subset)
        num_clients: Number of clients to create partitions for
        alpha: Dirichlet concentration parameter - controls skew
               alpha→0: extreme skew, each client gets mostly one class
               alpha→∞: balanced distribution (IID)
    
    Returns:
        A dictionary mapping client IDs to dataset subsets
    """
    # Extract targets based on dataset type
    if hasattr(dataset, 'targets'):
        # Regular dataset with targets attribute
        targets = np.array(dataset.targets)
    elif isinstance(dataset, Subset):
        # Handle Subset case
        if hasattr(dataset.dataset, 'targets'):
            # If the underlying dataset has targets attribute
            original_targets = dataset.dataset.targets
            if isinstance(original_targets, torch.Tensor):
                original_targets = original_targets.numpy()
            elif not isinstance(original_targets, np.ndarray):
                original_targets = np.array(original_targets)
            
            # Extract only the targets for the indices in the subset
            targets = original_targets[dataset.indices]
        else:
            # If the dataset doesn't have a targets attribute, try to get targets from __getitem__
            targets = []
            for i in range(len(dataset)):
                _, target = dataset[i]
                targets.append(target)
            targets = np.array(targets)
    else:
        # Last resort: try to extract targets by iterating through the dataset
        targets = []
        for i in range(len(dataset)):
            _, target = dataset[i]
            targets.append(target)
        targets = np.array(targets)

    classes = np.unique(targets)
    idx_by_class = {c: np.where(targets == c)[0] for c in classes}

    client_indices = [[] for _ in range(num_clients)]

    for c in classes:
        # draw class proportions
        props = np.random.dirichlet(alpha * np.ones(num_clients))
        props = (props * len(idx_by_class[c])).astype(int)
        
        # If there's rounding error, adjust last partition size
        props[-1] = len(idx_by_class[c]) - props[:-1].sum()

        # split indices
        np.random.shuffle(idx_by_class[c])
        start = 0
        for cid, cnt in enumerate(props):
            client_indices[cid].extend(idx_by_class[c][start:start+cnt])
            start += cnt

    # If working with a Subset, we need to remap the indices
    if isinstance(dataset, Subset):
        for cid in range(num_clients):
            # Map back to the original dataset indices
            client_indices[cid] = [dataset.indices[i] for i in client_indices[cid]]
            client_dataset = Subset(dataset.dataset, client_indices[cid])
            client_datasets = {cid: client_dataset for cid, idx in enumerate(client_indices)}
    else:
        client_datasets = {cid: Subset(dataset, idx) for cid, idx in enumerate(client_indices)}

    return client_datasets 