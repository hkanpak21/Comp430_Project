import torch
import torch.nn as nn
import torch.optim as optim
import logging

class MainServer:
    """Main Server responsible for the server-side model part and split computation."""
    def __init__(self, server_model_part: nn.Module, device, optimizer_name='sgd', lr=0.01):
        self.model_part = server_model_part.to(device)
        self.device = device

        if optimizer_name.lower() == 'sgd':
            self.optimizer = optim.SGD(self.model_part.parameters(), lr=lr)
        elif optimizer_name.lower() == 'adam':
            self.optimizer = optim.Adam(self.model_part.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        self.criterion = nn.CrossEntropyLoss()
        logging.info("MainServer initialized.")

    def process_batch(self, all_smashed_data: list, all_targets: list):
        """Processes a combined batch from clients, updates server model, returns gradients.

        Args:
            all_smashed_data (list): List of smashed data tensors (from one step) from participating clients.
            all_targets (list): List of corresponding target tensors.

        Returns:
            tuple: (average_loss_for_batch, list_of_gradients_for_clients)
                   Returns (0.0, []) if input is empty.
        """
        self.model_part.train() # Set server model to train mode
        self.optimizer.zero_grad()

        if not all_smashed_data or not all_targets:
            logging.warning("MainServer received no smashed data or targets. Skipping step.")
            return 0.0, []
        
        # Combine data from clients for this server step
        try:
            combined_smashed = torch.cat(all_smashed_data, dim=0).to(self.device)
            combined_targets = torch.cat(all_targets, dim=0).to(self.device)
        except Exception as e:
             logging.error(f"MainServer error during torch.cat: {e}")
             logging.error(f" Smashed data list lengths: {[d.shape for d in all_smashed_data]}")
             logging.error(f" Target list lengths: {[t.shape for t in all_targets]}")
             # Attempt to filter out potential None or zero-sized tensors
             filtered_smashed = [d.to(self.device) for d in all_smashed_data if d is not None and d.numel() > 0]
             filtered_targets = [t.to(self.device) for t in all_targets if t is not None and t.numel() > 0]
             if not filtered_smashed or not filtered_targets or len(filtered_smashed) != len(filtered_targets):
                 logging.error("Cannot recover from concatenation error. Skipping batch.")
                 return 0.0, []
             combined_smashed = torch.cat(filtered_smashed, dim=0)
             combined_targets = torch.cat(filtered_targets, dim=0)
             logging.warning("Recovered from concatenation error by filtering.")


        num_samples = combined_smashed.size(0)
        if num_samples == 0:
            logging.warning("MainServer: Combined batch size is 0 after concatenation. Skipping.")
            return 0.0, []

        # Ensure smashed data requires grad for server backward pass
        combined_smashed.requires_grad_(True)
        # Explicitly retain grad if it's not a leaf tensor (might be needed)
        if not combined_smashed.is_leaf:
             combined_smashed.retain_grad()

        # Server forward pass
        outputs = self.model_part(combined_smashed)

        # Calculate loss
        loss = self.criterion(outputs, combined_targets)

        # Server backward pass (calculates grads for server params AND smashed data)
        loss.backward()

        # Get gradients to send back to clients (gradient of loss w.r.t smashed data)
        # Detach and clone to prevent further modifications and move to CPU
        smashed_grads_combined = None
        if combined_smashed.grad is not None:
             smashed_grads_combined = combined_smashed.grad.detach().clone().cpu()
        else:
             logging.error("MainServer: combined_smashed.grad is None after loss.backward()! Cannot send gradients to clients.")
             # This indicates a problem in the graph or model structure
             # Send list of Nones of correct shape if possible, otherwise empty list
             gradients_to_clients = [None] * len(all_smashed_data)
             avg_loss = loss.item()
             # Still step the optimizer if loss was computed?
             # self.optimizer.step() # Maybe skip if grads are wrong
             return avg_loss, gradients_to_clients

        # Split gradients back based on original batch sizes from clients
        gradients_to_clients = []
        start_idx = 0
        for client_smashed_data in all_smashed_data:
            original_batch_size = client_smashed_data.size(0)
            end_idx = start_idx + original_batch_size
            if end_idx > smashed_grads_combined.size(0):
                 logging.error(f"Error splitting gradients: end_idx {end_idx} exceeds combined grad size {smashed_grads_combined.size(0)}")
                 # Append None or handle error appropriately
                 gradients_to_clients.append(None)
            else:
                 gradients_to_clients.append(smashed_grads_combined[start_idx:end_idx])
            start_idx = end_idx

        # Server optimizer step (updates server-side model weights)
        self.optimizer.step()

        avg_loss = loss.item() # Return loss per batch for monitoring
        # logging.debug(f"MainServer processed batch. Loss: {avg_loss:.4f}")
        return avg_loss, gradients_to_clients

    def get_server_model(self):
         """Returns the server-side model part (e.g., for evaluation)."""
         return self.model_part

# Remove the old evaluate method - evaluation should combine models externally 