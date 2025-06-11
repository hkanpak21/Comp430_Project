import torch
import torch.nn as nn
import torch.optim as optim # Import optim
from collections import OrderedDict
from typing import List, Dict
import numpy as np

from .aggregation import federated_averaging_gradients, federated_averaging

class FedServer:
    """
    Federated Server (FedServer) in SFLV1.
    Manages the global client-side model (WC) and aggregates client updates.
    """
    def __init__(self, client_model: nn.Module, config: dict, device: torch.device):
        """
        Args:
            client_model: An instance of the client-side model (WC) architecture.
            config: Configuration dictionary.
            device: The torch device ('cpu' or 'cuda').
        """
        self.client_model = client_model.to(device) # Holds the global WC parameters
        self.config = config
        self.device = device
        self.optimizer = self._create_optimizer() # Add optimizer for WC
        self._client_updates = [] # Stores received client updates (gradients or models) in a round

        # Adaptive DP: Track validation loss and noise scale
        self._current_sigma = config['dp_noise']['initial_sigma']
        self._validation_losses = []  # Store recent validation losses
        self._noise_decay_patience = config['dp_noise']['noise_decay_patience']
        self._adaptive_noise_decay_factor = config['dp_noise']['adaptive_noise_decay_factor']
        self._criterion = nn.CrossEntropyLoss()
        self._sigma_history = [self._current_sigma]  # Track sigma history

    def _create_optimizer(self) -> optim.Optimizer:
        """Creates the optimizer for the global client-side model (WC)."""
        lr = self.config.get('lr', 0.01)
        optimizer_name = self.config.get('optimizer', 'SGD').lower()

        if optimizer_name == 'sgd':
            # Consider adding momentum here if needed based on config
            return optim.SGD(self.client_model.parameters(), lr=lr)
        elif optimizer_name == 'adam':
            return optim.Adam(self.client_model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def get_client_model_params(self) -> OrderedDict:
        """Returns the state dictionary of the current global client model (WC)."""
        return self.client_model.state_dict()

    def receive_client_update(self, client_update: Dict[str, torch.Tensor]):
        """
        Receives and stores an update (typically gradients) from a client.

        Args:
            client_update: A dictionary containing the gradients or model parameters from a client.
                           Gradients must be detached and moved to the correct device if necessary
                           before being sent here.
        """
        # Ensure updates are on the correct device and detached
        processed_update = OrderedDict()
        for name, param in client_update.items():
            processed_update[name] = param.detach().clone().to(self.device)
        self._client_updates.append(processed_update)

    def evaluate_validation_loss(self, validation_loader) -> float:
        """
        Evaluates the current model on the validation set.
        """
        self.client_model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for data, labels in validation_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.client_model(data)
                loss = self._criterion(outputs, labels)
                total_loss += loss.item()
        return total_loss / len(validation_loader)

    def evaluate_metrics(self, validation_loader, main_server=None) -> tuple:
        """
        Evaluates and prints both validation loss and accuracy.
        If main_server is provided, evaluates the complete model (client+server).
        Otherwise, just evaluates the client part (which will show poor metrics).
        
        Args:
            validation_loader: DataLoader for validation set
            main_server: Optional MainServer instance to evaluate complete model
            
        Returns:
            tuple: (validation_loss, accuracy)
        """
        self.client_model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Check if we can evaluate complete model
        evaluate_complete = main_server is not None
        
        if evaluate_complete:
            # Get server model
            server_model = main_server.get_server_model()
            server_model.eval()
        
        with torch.no_grad():
            for data, labels in validation_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                
                # Forward pass through client model
                client_outputs = self.client_model(data)
                
                if evaluate_complete:
                    # Complete forward pass through server model
                    outputs = server_model(client_outputs)
                    loss = main_server.get_criterion()(outputs, labels)
                    
                    # Calculate accuracy
                    _, predicted = torch.max(outputs.data, 1)
                else:
                    # Just use client outputs (which aren't actual predictions)
                    outputs = client_outputs
                    # Using CrossEntropyLoss directly on activations doesn't make sense
                    # but keeping it to maintain backward compatibility
                    loss = self._criterion(outputs, labels)
                    
                    # This won't be meaningful but keeping for compatibility
                    _, predicted = torch.max(outputs.data, 1)
                
                total_loss += loss.item()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(validation_loader)
        accuracy = 100 * correct / total
        
        print(f"\nValidation Metrics:")
        if evaluate_complete:
            print(f"Complete Model - Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
        else:
            print(f"Client Model Only (incomplete) - Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
            print(f"Warning: These metrics are for the client model part only and don't reflect actual performance.")
        
        return avg_loss, accuracy

    def _update_noise_scale(self, validation_loss: float):
        """
        Updates the noise scale based on validation loss trend.
        """
        self._validation_losses.append(validation_loss)
        
        # Check if we have enough history to make a decision
        if len(self._validation_losses) < self._noise_decay_patience + 1:
            return
        
        # Check if loss has been decreasing for the required number of rounds
        recent_losses = self._validation_losses[-self._noise_decay_patience:]
        is_decreasing = all(recent_losses[i] > recent_losses[i+1] 
                          for i in range(len(recent_losses)-1))
        
        if is_decreasing:
            # Decrease noise scale
            self._current_sigma *= self._adaptive_noise_decay_factor
            print(f"FedServer: Loss decreasing for {self._noise_decay_patience} rounds. "
                  f"Updated noise scale to {self._current_sigma:.4f}")
            # Track sigma change
            self._sigma_history.append(self._current_sigma)

    def get_current_sigma(self) -> float:
        """Returns the current noise scale sigma_t."""
        return self._current_sigma

    def aggregate_updates(self, validation_loader=None, main_server=None):
        """
        Aggregates the received client updates using FedAvg and updates the global client model (WC)
        using its optimizer.
        Clears the stored updates after aggregation.
        
        Args:
            validation_loader: Optional dataloader for validation
            main_server: Optional MainServer instance to evaluate complete model
        """
        if not self._client_updates:
            print("FedServer: No client updates received for aggregation.")
            return

        # Assume updates are gradients based on SFLV1 flow (noisy ∇WCk,t)
        averaged_gradients = federated_averaging_gradients(self._client_updates)

        # Update the global client model parameters using the averaged gradients via the optimizer
        self.optimizer.zero_grad()
        with torch.no_grad(): # Manually assign gradients
            for name, param in self.client_model.named_parameters():
                if name in averaged_gradients:
                    if param.grad is None:
                        param.grad = torch.zeros_like(param)
                    param.grad.copy_(averaged_gradients[name])
                else:
                    # This case might indicate an issue - all client params should ideally get grads
                    print(f"Warning: Averaged gradient for '{name}' not found during FedServer update.")
                    if param.grad is not None:
                        param.grad.zero_() # Zero out if it exists but wasn't in average

        self.optimizer.step() # Update parameters using assigned gradients

        # Update noise scale and evaluate metrics if validation loader is provided
        if validation_loader is not None:
            validation_loss, accuracy = self.evaluate_metrics(validation_loader, main_server)
            self._update_noise_scale(validation_loss)

        # Clear updates for the next round
        self._client_updates = []

        # Count how many parameter groups were updated by the optimizer
        updated_param_count = sum(1 for group in self.optimizer.param_groups for p in group['params'])
        print(f"FedServer: Aggregated client updates and updated WC model ({updated_param_count} params) via optimizer.")

    def get_client_model(self) -> nn.Module:
        """Returns the current global client model instance."""
        return self.client_model 