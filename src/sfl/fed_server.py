import torch
import torch.nn as nn
from collections import OrderedDict
import logging
from copy import deepcopy

class FedServer:
    """Federated Server responsible for aggregating client-side model updates."""
    def __init__(self, initial_client_model: nn.Module):
        self.global_client_model = deepcopy(initial_client_model)
        logging.info("FedServer initialized.")

    def aggregate_models(self, client_models_states: list, client_sample_counts: list):
        """Aggregates client model state dicts using Federated Averaging.

        Args:
            client_models_states: A list of state_dict objects from clients.
            client_sample_counts: A list of the number of samples each client used.
        """
        if not client_models_states or not client_sample_counts:
            logging.warning("FedServer received no models or counts for aggregation. Skipping.")
            return
        
        if len(client_models_states) != len(client_sample_counts):
             raise ValueError("Number of client models and sample counts must match.")

        total_samples = sum(client_sample_counts)
        if total_samples == 0:
             logging.warning("FedServer: Total samples for aggregation is 0. Skipping aggregation.")
             return

        global_state_dict = self.global_client_model.state_dict()
        aggregated_state_dict = OrderedDict()

        logging.info(f"FedServer aggregating models from {len(client_models_states)} clients with {total_samples} total samples.")

        # Initialize aggregated state dict with zeros
        for key in global_state_dict.keys():
            aggregated_state_dict[key] = torch.zeros_like(global_state_dict[key], dtype=torch.float32)

        # Perform weighted averaging
        for i, state_dict in enumerate(client_models_states):
            weight = client_sample_counts[i] / total_samples
            for key in global_state_dict.keys():
                if key in state_dict:
                     # Ensure tensors are float for accumulation
                    aggregated_state_dict[key] += state_dict[key].float() * weight
                else:
                    logging.warning(f"Key {key} not found in state dict from client {i}. Skipping for this client.")


        # Load the aggregated state dict into the global model
        self.global_client_model.load_state_dict(aggregated_state_dict)
        logging.info("FedServer finished model aggregation.")

    def get_global_model_state(self):
        """Returns the state dict of the current global client-side model."""
        return self.global_client_model.state_dict()

    def get_global_model(self):
        """Returns the current global client-side model itself (e.g., for evaluation)."""
        return self.global_client_model 