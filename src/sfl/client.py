import torch
import torch.nn as nn
import torch.optim as optim
from opacus.utils.batch_memory_manager import BatchMemoryManager

class SFLClient:
    def __init__(self, client_id, model_part, dataloader, optimizer_name, lr, device, local_epochs):
        self.id = client_id
        self.model_part = model_part.to(device)
        self.dataloader = dataloader
        self.device = device
        self.local_epochs = local_epochs

        if optimizer_name.lower() == 'sgd':
            self.optimizer = optim.SGD(self.model_part.parameters(), lr=lr)
        elif optimizer_name.lower() == 'adam':
            self.optimizer = optim.Adam(self.model_part.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        self.criterion = nn.CrossEntropyLoss() # Or appropriate loss
        self.smashed_data = None

    def train_step(self, server_gradients):
        """Performs one local training step (forward, backward, step)."""
        self.model_part.train()
        running_loss = 0.0
        num_batches = 0

        # Handle potential gradients from the previous round (if applicable)
        if server_gradients is not None:
            self.smashed_data.backward(server_gradients.to(self.device))
            self.optimizer.step()

        # Perform local epochs
        for epoch in range(self.local_epochs):
            # print(f"Client {self.id}, Local Epoch {epoch+1}/{self.local_epochs}")
            if isinstance(self.dataloader, BatchMemoryManager):
                 # Special handling if using Opacus BatchMemoryManager for DP
                 # It expects max_physical_batch_size, but we pass the whole logical batch
                 # This loop iterates over logical batches; BMM handles physical ones.
                batch_count_in_epoch = 0
                for data, target in self.dataloader: # BMM yields logical batches
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()
                    # Forward pass up to the cut layer
                    self.smashed_data = self.model_part(data)
                    # Stop computation here, detach and send to server
                    # Actual loss calculation and backprop happens after server interaction
                    # We return the smashed data for the server
                    batch_count_in_epoch += 1
                    # In a real SFL system, we would pause here and wait for server grads
                    # For simulation, we return smashed data and expect grads later
                    yield self.smashed_data.detach().clone().cpu(), target.cpu() # Send detached data and target
                # print(f"Client {self.id}, Local Epoch {epoch+1}, Batches: {batch_count_in_epoch}")
            else:
                # Standard DataLoader iteration
                batch_count_in_epoch = 0
                for data, target in self.dataloader:
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()
                    self.smashed_data = self.model_part(data)
                    batch_count_in_epoch += 1
                    # Yield smashed data and target
                    yield self.smashed_data.detach().clone().cpu(), target.cpu()
                # print(f"Client {self.id}, Local Epoch {epoch+1}, Batches: {batch_count_in_epoch}")

        # Return None after local epochs are done to signal completion for this round
        yield None, None


    def apply_gradients(self, gradients):
        """Applies gradients received from the server."""
        if gradients is not None and self.smashed_data is not None:
            try:
                self.smashed_data.backward(gradients.to(self.device))
                self.optimizer.step()
            except RuntimeError as e:
                print(f"Client {self.id}: Error during backward/step: {e}")
                # Optional: Add more debugging info, e.g., check shapes
                print(f"Smashed data shape: {self.smashed_data.shape}, requires_grad: {self.smashed_data.requires_grad}")
                print(f"Gradients shape: {gradients.shape}")
                raise e
        self.smashed_data = None # Clear after use 