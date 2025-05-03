import torch
import torch.nn as nn
import torch.optim as optim
import logging
from copy import deepcopy

class Client:
    """Client entity in Split Federated Learning."""
    def __init__(self, client_id, initial_model_part: nn.Module, dataloader, 
                 optimizer_name, lr, device, local_epochs):
        self.id = client_id
        self.model_part = deepcopy(initial_model_part).to(device) # Keep local copy
        self.dataloader = dataloader
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.device = device
        self.local_epochs = local_epochs
        
        self._initialize_optimizer() 
        self.criterion = nn.CrossEntropyLoss() # Although loss is calculated on server, client might need it?
                                             # Keep it for now, maybe useful for future extensions.
                                             # Or remove if strictly not needed.

        self.smashed_data_cache = {} # Cache smashed data for backward pass {batch_idx: data}
        self.data_iterator = None
        self.current_epoch = 0
        self.current_batch_idx = 0
        self.is_done = False

        self.sample_count = len(dataloader.dataset) if dataloader and hasattr(dataloader, 'dataset') else 0
        logging.info(f"Client {self.id} initialized with {self.sample_count} samples. Model part on {self.device}.")


    def _initialize_optimizer(self):
        """Initializes the optimizer."""
        if self.optimizer_name.lower() == 'sgd':
            self.optimizer = optim.SGD(self.model_part.parameters(), lr=self.lr)
        elif self.optimizer_name.lower() == 'adam':
            self.optimizer = optim.Adam(self.model_part.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
        logging.debug(f"Client {self.id}: Optimizer initialized ({self.optimizer_name}, lr={self.lr}).")

    def set_model(self, state_dict):
        """Sets the client's model part based on a state dictionary from FedServer."""
        try:
            self.model_part.load_state_dict(state_dict)
            logging.debug(f"Client {self.id}: Model updated from FedServer.")
        except RuntimeError as e:
            logging.error(f"Client {self.id}: Error loading state dict: {e}")
            # Log keys to help debugging
            logging.error(f" State Dict Keys: {list(state_dict.keys())}")
            logging.error(f" Model Keys: {list(self.model_part.state_dict().keys())}")
            raise e
        # Re-initialize optimizer state if needed (e.g., if Adam state is tied to specific params)
        # For simplicity, we re-init optimizer completely. Consider state loading if needed.
        self._initialize_optimizer()


    def get_model_state(self):
        """Returns the state dict of the client's model part."""
        return self.model_part.state_dict()

    def get_sample_count(self):
        """Returns the number of training samples this client has."""
        return self.sample_count

    def reset_training(self):
        """Resets the client's training state for a new round."""
        self.current_epoch = 0
        self.current_batch_idx = 0
        self.is_done = False
        self.smashed_data_cache.clear()
        # Ensure dataloader gives fresh data if possible (depends on dataloader impl)
        if self.dataloader:
            self.data_iterator = iter(self.dataloader)
        else:
            self.data_iterator = None
            self.is_done = True # No data, done immediately
        logging.debug(f"Client {self.id}: Training state reset.")

    def local_step(self):
        """Performs one step of local computation (forward pass for one batch).
        
        Returns:
            tuple: (smashed_data, targets, batch_idx, is_done)
                   Returns (None, None, -1, True) if training is finished for this round.
                   batch_idx is used to match gradients later.
        """
        if self.is_done or self.data_iterator is None:
             return None, None, -1, True

        try:
            data, target = next(self.data_iterator)
            current_batch_idx_for_step = self.current_batch_idx # Store before incrementing
            self.current_batch_idx += 1

        except StopIteration:
            # End of current epoch
            self.current_epoch += 1
            if self.current_epoch >= self.local_epochs:
                # Finished all local epochs for this round
                self.is_done = True
                logging.debug(f"Client {self.id}: Finished local epochs ({self.local_epochs}).")
                return None, None, -1, True
            else:
                # Start next epoch
                logging.debug(f"Client {self.id}: Starting local epoch {self.current_epoch + 1}")
                self.current_batch_idx = 0
                self.data_iterator = iter(self.dataloader)
                # Recursively call to get the first batch of the new epoch
                return self.local_step()

        # --- Forward Pass ---
        self.model_part.train() # Ensure model is in training mode
        self.optimizer.zero_grad() # Zero gradients before forward pass
        
        data, target = data.to(self.device), target.to(self.device)
        
        # Perform client-side forward pass
        smashed_data = self.model_part(data)
        
        # Cache smashed data for backward pass, using the batch index as key
        # Detach required because we will use it in backward later, don't want graph issues
        self.smashed_data_cache[current_batch_idx_for_step] = smashed_data # Don't detach yet, need graph for backward
        
        # Return detached data for sending (no grad needed for MainServer input initially)
        # Send targets as well
        logging.debug(f"Client {self.id}: Forward pass complete for batch {current_batch_idx_for_step}. Smashed shape: {smashed_data.shape}")
        return smashed_data.detach().clone().cpu(), target.clone().cpu(), current_batch_idx_for_step, False

    def apply_gradients(self, gradients, batch_idx):
        """Applies gradients received from the MainServer for a specific batch."""
        if gradients is None:
            logging.warning(f"Client {self.id}: Received None gradients for batch {batch_idx}. Skipping backward/step.")
            # Clean up cache if gradients are None
            if batch_idx in self.smashed_data_cache:
                del self.smashed_data_cache[batch_idx]
            return

        if batch_idx not in self.smashed_data_cache:
             logging.error(f"Client {self.id}: No cached smashed data found for batch {batch_idx}. Cannot apply gradients.")
             return

        # Retrieve cached smashed data that requires grad
        smashed_data = self.smashed_data_cache[batch_idx]
        
        if not smashed_data.requires_grad:
             logging.error(f"Client {self.id}: Cached smashed data for batch {batch_idx} does not require grad! Cannot perform backward.")
             del self.smashed_data_cache[batch_idx] # Clean cache
             return

        try:
            # Perform backward pass using gradients from server
            logging.debug(f"Client {self.id}: Applying gradients for batch {batch_idx}. Grad shape: {gradients.shape}")
            smashed_data.backward(gradients.to(self.device))

            # --- Parameter Update ---
            # No DP noise in this version
            self.optimizer.step()
            logging.debug(f"Client {self.id}: Optimizer step complete for batch {batch_idx}.")

        except RuntimeError as e:
            logging.error(f"Client {self.id}: Error during backward/step for batch {batch_idx}: {e}", exc_info=True)
            logging.error(f" Smashed data shape: {smashed_data.shape}, requires_grad: {smashed_data.requires_grad}")
            logging.error(f" Received Gradients shape: {gradients.shape}")
            # Don't raise, allow training to continue if possible, but log error
        finally:
             # Clean up cache for this batch index after backward/step
             if batch_idx in self.smashed_data_cache:
                 del self.smashed_data_cache[batch_idx]
             # Gradients are zeroed before the next forward pass in local_step


# Removed DP related methods (get_feedback_metric, calculate_new_dp_params) 