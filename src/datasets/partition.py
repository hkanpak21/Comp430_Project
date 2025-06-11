import numpy as np
from torch.utils.data import Subset

def partition_data_dirichlet(dataset, num_clients, alpha):
    """
    Partitions the data using a Dirichlet distribution to create non-IID data splits.
    
    Args:
        dataset: The dataset to partition (must have a 'targets' attribute)
        num_clients: Number of clients to create partitions for
        alpha: Dirichlet concentration parameter - controls skew
               alpha→0: extreme skew, each client gets mostly one class
               alpha→∞: balanced distribution (IID)
    
    Returns:
        A dictionary mapping client IDs to dataset subsets
    """
    targets = np.array(dataset.targets)
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

    return {cid: Subset(dataset, idx) for cid, idx in enumerate(client_indices)} 