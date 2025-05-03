import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy

class SFLServer:
    def __init__(self, model_part, test_loader, optimizer_name, lr, device):
        self.model_part = model_part.to(device)
        self.test_loader = test_loader
        self.device = device

        if optimizer_name.lower() == 'sgd':
            self.optimizer = optim.SGD(self.model_part.parameters(), lr=lr)
        elif optimizer_name.lower() == 'adam':
            self.optimizer = optim.Adam(self.model_part.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        self.criterion = nn.CrossEntropyLoss() # Or NLLLoss if model output is log_softmax

    def train_step(self, all_smashed_data, all_targets):
        """Performs server-side training step: forward, loss, backward, step.

        Args:
            all_smashed_data (list): List of smashed data tensors from participating clients.
            all_targets (list): List of corresponding target tensors from participating clients.

        Returns:
            tuple: (average_loss, list_of_gradients_for_clients)
        """
        self.model_part.train()
        self.optimizer.zero_grad()

        if not all_smashed_data:
            return 0.0, []

        gradients_to_clients = []
        total_loss = 0.0
        num_samples = 0

        # Process batches from all clients contributing to this server step
        # Assume all_smashed_data and all_targets correspond to batches from clients in this step
        combined_smashed = torch.cat(all_smashed_data, dim=0).to(self.device)
        combined_targets = torch.cat(all_targets, dim=0).to(self.device)

        # Ensure smashed data requires grad for server backward pass
        combined_smashed.requires_grad_(True)

        # Server forward pass
        outputs = self.model_part(combined_smashed)

        # Calculate loss
        loss = self.criterion(outputs, combined_targets)
        total_loss += loss.item() * combined_smashed.size(0)
        num_samples += combined_smashed.size(0)

        # Server backward pass
        loss.backward()

        # Get gradients to send back to clients (gradient of loss w.r.t smashed data)
        # Detach and clone to prevent further modifications
        smashed_grads = combined_smashed.grad.detach().clone().cpu()

        # Split gradients back based on original batch sizes
        start_idx = 0
        for client_smashed_data in all_smashed_data:
            end_idx = start_idx + client_smashed_data.size(0)
            gradients_to_clients.append(smashed_grads[start_idx:end_idx])
            start_idx = end_idx

        # Server optimizer step (updates server-side model weights)
        self.optimizer.step()

        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        return avg_loss, gradients_to_clients

    def evaluate(self, client_model_part):
        """Evaluates the combined model on the test set."""
        self.model_part.eval()
        client_model_part.eval()
        correct = 0
        total = 0
        test_loss = 0.0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)

                # Client forward pass (on device)
                smashed_data = client_model_part(data)

                # Server forward pass (on device)
                outputs = self.model_part(smashed_data)

                loss = self.criterion(outputs, target)
                test_loss += loss.item() * data.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100.0 * correct / total if total > 0 else 0.0
        avg_test_loss = test_loss / total if total > 0 else 0.0
        return accuracy, avg_test_loss 