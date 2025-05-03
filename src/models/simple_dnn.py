import torch
import torch.nn as nn
import logging
from collections import OrderedDict

class SimpleDNN(nn.Module):
    """A simple fully-connected neural network for MNIST."""
    def __init__(self, hidden_layers=[128, 64], num_classes=10):
        super(SimpleDNN, self).__init__()
        layers = OrderedDict()
        input_dim = 28 * 28
        
        # Input layer
        layers['fc1'] = nn.Linear(input_dim, hidden_layers[0])
        layers['relu1'] = nn.ReLU()
        layers['flatten'] = nn.Flatten() # Ensure input is flattened

        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            layers[f'fc{i+2}'] = nn.Linear(hidden_layers[i], hidden_layers[i+1])
            layers[f'relu{i+2}'] = nn.ReLU()

        # Output layer
        layers['fc_out'] = nn.Linear(hidden_layers[-1], num_classes)
        # Note: No final activation (like Softmax or LogSoftmax) included here
        # nn.CrossEntropyLoss expects raw logits.

        self.network = nn.Sequential(layers)

    def forward(self, x):
        # Flatten the image
        x = x.view(x.size(0), -1) # Flatten the input image
        return self.network(x)

def split_model(model: nn.Module, cut_layer_idx: int):
    """Splits a sequential model into client and server parts at a given index.

    Args:
        model: The nn.Sequential model to split.
        cut_layer_idx: The index of the layer *after* which to cut.
                       All layers up to and including this index are on the client.
                       Layers after this index are on the server.

    Returns:
        A tuple: (client_model_part, server_model_part)
    """
    if not isinstance(model.network, nn.Sequential):
         raise TypeError("Model splitting currently only supports models with a top-level nn.Sequential named 'network'")

    all_layers = list(model.network.named_children())
    if cut_layer_idx < 0 or cut_layer_idx >= len(all_layers):
        raise ValueError(f"cut_layer_idx ({cut_layer_idx}) is out of bounds for model with {len(all_layers)} layers.")

    logging.info(f"Splitting model after layer {cut_layer_idx}: '{all_layers[cut_layer_idx][0]}'")

    client_layers = OrderedDict(all_layers[:cut_layer_idx + 1])
    server_layers = OrderedDict(all_layers[cut_layer_idx + 1:])

    if not server_layers:
        raise ValueError("Cannot split after the last layer - server model would be empty.")

    # Important: The client part needs to handle the initial flattening if it wasn't done explicitly before Sequential
    # In our SimpleDNN, Flatten is layer 0, so it will always be on the client unless cut_layer_idx = -1 (which is invalid)
    # If the first layer isn't Flatten, the client needs to handle it.
    class ClientWrapper(nn.Module):
        def __init__(self, client_seq):
            super().__init__()
            self.client_part = client_seq
        def forward(self, x):
             # Check if input needs flattening (simple heuristic)
             if x.ndim > 2 and not isinstance(self.client_part[0], nn.Flatten):
                 x = x.view(x.size(0), -1)
             return self.client_part(x)

    client_model_part = ClientWrapper(nn.Sequential(client_layers))
    server_model_part = nn.Sequential(server_layers)

    # Log parts
    logging.debug("Client Model Part:")
    logging.debug(client_model_part)
    logging.debug("Server Model Part:")
    logging.debug(server_model_part)

    return client_model_part, server_model_part 