import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from typing import Dict, Tuple, List

from .aggregation import federated_averaging_gradients

class MainServer:
    """
    Main Server in SFLV1.
    Manages the server-side model (WS), performs server-side computations,
    and aggregates WS updates.
    """
    def __init__(self, server_model: nn.Module, config: dict, device: torch.device):
        """
        Args:
            server_model: An instance of the server-side model (WS) architecture.
            config: Configuration dictionary.
            device: The torch device ('cpu' or 'cuda').
        """
        self.server_model = server_model.to(device)
        self.config = config
        self.device = device
        self.optimizer = self._create_optimizer()
        self.criterion = nn.CrossEntropyLoss() # Assuming classification task

        # Store intermediate values needed for backward pass and aggregation
        self._client_activations: Dict[int, torch.Tensor] = {} # {client_id: A_k,t}
        self._client_labels: Dict[int, torch.Tensor] = {}    # {client_id: Y_k}
        self._server_gradients: List[Dict[str, torch.Tensor]] = [] # Stores ∇WS_k from each client

    def _create_optimizer(self) -> optim.Optimizer:
        """Creates the optimizer for the server-side model (WS)."""
        lr = self.config.get('lr', 0.01)
        optimizer_name = self.config.get('optimizer', 'SGD').lower()

        if optimizer_name == 'sgd':
            return optim.SGD(self.server_model.parameters(), lr=lr)
        elif optimizer_name == 'adam':
            return optim.Adam(self.server_model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def receive_client_data(self, client_id: int, activations: torch.Tensor, labels: torch.Tensor):
        """
        Receives smashed data (activations) and labels from a client.

        Args:
            client_id: The ID of the sending client.
            activations: The (potentially noisy) output tensor (Ak,t) from the client's model part.
            labels: The corresponding labels (Yk) for the batch.
        """
        # Ensure data is on the correct device and requires grad for server backward pass
        self._client_activations[client_id] = activations.detach().clone().to(self.device).requires_grad_(True)
        self._client_labels[client_id] = labels.clone().to(self.device)
        # print(f"MainServer: Received data from Client {client_id}. Activations shape: {activations.shape}, Labels shape: {labels.shape}")

    def forward_backward_pass(self, client_id: int) -> torch.Tensor:
        """
        Performs the forward and backward pass for a specific client's data
        on the server-side model (WS).

        Args:
            client_id: The ID of the client whose data should be processed.

        Returns:
            The gradient of the loss with respect to the client's activations (∇Ak,t),
            to be sent back to the client.
        """
        if client_id not in self._client_activations:
            raise ValueError(f"MainServer: No activations found for client {client_id}. Call receive_client_data first.")

        activations = self._client_activations[client_id]
        labels = self._client_labels[client_id]

        # Zero gradients for the server model parameters for this specific client pass
        # Note: We aggregate gradients later, zeroing here is standard practice per client batch
        self.optimizer.zero_grad()

        # Forward pass through server model (WS)
        outputs = self.server_model(activations)
        loss = self.criterion(outputs, labels)

        # Backward pass to compute gradients (both ∇WS and ∇Ak,t)
        loss.backward()

        # Store the gradients of the server model parameters (∇WS_k) for this client
        # We need to clone them as they will be overwritten/zeroed in subsequent steps
        server_grads_k = OrderedDict()
        for name, param in self.server_model.named_parameters():
            if param.grad is not None:
                server_grads_k[name] = param.grad.detach().clone()
            else:
                # This might happen for layers without parameters or if computation graph is detached
                 print(f"Warning: No gradient for server parameter '{name}' for client {client_id}.")
        if server_grads_k: # Only store if gradients were computed
            self._server_gradients.append(server_grads_k)

        # Get the gradient w.r.t the activations (∇Ak,t)
        activation_grad = activations.grad.detach().clone() if activations.grad is not None else None
        if activation_grad is None:
             print(f"Warning: Activation gradient (∇Ak,t) is None for client {client_id}. Check model structure and requires_grad.")
             # Return a zero tensor of the expected shape if grad is None, to avoid crashing client
             # This might indicate an issue elsewhere (e.g., no path back from loss to activation)
             activation_grad = torch.zeros_like(activations)

        # print(f"MainServer: Completed forward/backward for Client {client_id}. Loss: {loss.item():.4f}")

        # Return the activation gradient to the client
        return activation_grad

    def aggregate_and_update(self):
        """
        Aggregates the collected server-side gradients (∇WS_k) using FedAvg
        and performs an optimization step on the server model (WS).
        Clears stored activations, labels, and gradients after update.
        """
        if not self._server_gradients:
            print("MainServer: No server gradients collected for aggregation.")
            self.clear_round_data() # Clear any leftover client data
            return

        # Average the collected server gradients (∇WS_k)
        averaged_gradients = federated_averaging_gradients(self._server_gradients)

        # Manually set the gradients of the server model to the averaged gradients
        self.optimizer.zero_grad() # Zero gradients first
        with torch.no_grad():
            for name, param in self.server_model.named_parameters():
                if name in averaged_gradients:
                    if param.grad is None: # Initialize gradient tensor if needed
                         param.grad = torch.zeros_like(param)
                    param.grad.copy_(averaged_gradients[name])
                else:
                    # Might happen if a parameter didn't receive grad from any client
                    print(f"Warning: Averaged gradient for '{name}' not found during MainServer update.")
                    if param.grad is not None:
                        param.grad.zero_() # Ensure it's zero if no update

        # Perform the optimization step using the averaged gradients
        self.optimizer.step()

        print(f"MainServer: Aggregated {len(self._server_gradients)} server gradients and updated WS.")

        # Clear data for the next round
        self.clear_round_data()

    def clear_round_data(self):
        """Clears activations, labels, and gradients stored for the current round."""
        self._client_activations.clear()
        self._client_labels.clear()
        self._server_gradients.clear()

    def get_server_model(self) -> nn.Module:
        """Returns the current server model instance."""
        return self.server_model 

    def get_criterion(self) -> nn.Module:
        """Returns the criterion (loss function) used by the server."""
        return self.criterion 