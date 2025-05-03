import torch
from collections import OrderedDict
from typing import List, Dict

def federated_averaging(updates: List[OrderedDict]) -> OrderedDict:
    """
    Performs Federated Averaging (FedAvg) on a list of state dict updates.

    Args:
        updates: A list where each element is an OrderedDict representing
                 a model's state_dict or gradients (from model.named_parameters()).

    Returns:
        An OrderedDict representing the averaged state_dict or gradients.
    """
    if not updates:
        return OrderedDict()

    # Initialize averaged update with zeros, based on the structure of the first update
    averaged_update = OrderedDict()
    for key, tensor in updates[0].items():
        averaged_update[key] = torch.zeros_like(tensor)

    # Sum up all updates
    num_updates = len(updates)
    for update in updates:
        for key, tensor in update.items():
            if key in averaged_update:
                averaged_update[key] += tensor.detach().clone() # Use detach().clone() to avoid modifying originals
            else:
                # This shouldn't happen if all updates have the same structure
                print(f"Warning: Key '{key}' not found in initial update structure. Skipping.")

    # Divide by the number of updates to get the average
    for key in averaged_update:
        averaged_update[key] /= num_updates

    return averaged_update

def federated_averaging_gradients(gradients_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Performs Federated Averaging specifically on a list of gradients.
    Assumes gradients are stored as Dict[parameter_name, gradient_tensor].
    This is essentially the same as federated_averaging but with explicit typing for gradients.

    Args:
        gradients_list: A list where each element is a dictionary mapping parameter
                        names to their gradient tensors.

    Returns:
        A dictionary representing the averaged gradients.
    """
    if not gradients_list:
        return {}

    # Initialize averaged gradients with zeros, based on the first client's gradients
    averaged_gradients = OrderedDict()
    for key, tensor in gradients_list[0].items():
        averaged_gradients[key] = torch.zeros_like(tensor)

    # Sum up all gradients
    num_gradients = len(gradients_list)
    for gradients in gradients_list:
        for key, tensor in gradients.items():
            if key in averaged_gradients:
                # Ensure gradients are properly detached if they come from autograd graph
                averaged_gradients[key] += tensor.detach().clone()
            else:
                print(f"Warning: Gradient key '{key}' not found in initial structure. Skipping.")

    # Average the gradients
    for key in averaged_gradients:
        averaged_gradients[key] /= num_gradients

    return averaged_gradients 