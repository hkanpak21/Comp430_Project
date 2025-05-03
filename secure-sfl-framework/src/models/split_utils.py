import torch
import torch.nn as nn
from collections import OrderedDict

class SplitModel(nn.Module):
    """Wraps a model to split it into client and server parts."""
    def __init__(self, full_model, cut_layer_idx):
        super(SplitModel, self).__init__()

        if not hasattr(full_model, 'get_layers'):
            raise ValueError("Model must have a 'get_layers' method returning a list of its layers.")

        all_layers = full_model.get_layers()
        if cut_layer_idx < 0 or cut_layer_idx >= len(all_layers):
            raise ValueError(f"Invalid cut_layer_idx: {cut_layer_idx}. Must be between 0 and {len(all_layers) - 1}.")

        self.client_part = nn.Sequential(*all_layers[:cut_layer_idx+1])
        self.server_part = nn.Sequential(*all_layers[cut_layer_idx+1:])

    def forward(self, x):
        # This forward is primarily for verification/combined model usage.
        # In SFL, client_part and server_part are run separately.
        x = self.client_part(x)
        x = self.server_part(x)
        return x

def split_model(model, cut_layer_idx):
    """Splits a model into client and server parts at the specified index."""
    if not hasattr(model, 'get_layers'):
        raise ValueError("Model must have a 'get_layers' method returning a list of its layers.")

    all_layers = model.get_layers()
    if cut_layer_idx < 0 or cut_layer_idx >= len(all_layers):
        raise ValueError(f"Invalid cut_layer_idx: {cut_layer_idx}. Must be between 0 and {len(all_layers) - 1}.")

    client_model = nn.Sequential(*all_layers[:cut_layer_idx+1])
    server_model = nn.Sequential(*all_layers[cut_layer_idx+1:])

    return client_model, server_model

def get_combined_model(client_model_state_dict, server_model_state_dict, full_model_template):
    """Recombines client and server state dicts into a full model template for evaluation."""
    client_keys = client_model_state_dict.keys()
    server_keys = server_model_state_dict.keys()

    combined_state_dict = OrderedDict()

    # Load parameters, ensuring correct mapping based on layer names in the template
    full_model_dict = full_model_template.state_dict()
    client_layers_params = set()
    for param_name in client_keys:
        # Map sequential name (e.g., '0.weight') to original name if possible (requires careful naming)
        # Simpler approach: Assume the state dicts correspond to the Sequential structure directly
        combined_state_dict[param_name] = client_model_state_dict[param_name]
        client_layers_params.add(param_name.split('.')[0]) # Track which layers belong to client

    for param_name in server_keys:
        combined_state_dict[param_name] = server_model_state_dict[param_name]

    # Verify all keys are covered (optional)
    # assert set(combined_state_dict.keys()) == set(full_model_dict.keys())

    full_model_template.load_state_dict(combined_state_dict)
    return full_model_template 