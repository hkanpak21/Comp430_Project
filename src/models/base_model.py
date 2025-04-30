import torch.nn as nn

class SplitModelBase(nn.Module):
    """Base class to facilitate splitting a model."""
    def __init__(self):
        super().__init__()
        self._split_layer_name = None
        self._client_part = nn.Sequential()
        self._server_part = nn.Sequential()

    def _find_split_layer(self, split_layer_name):
        """Finds the split layer and separates the model parts."""
        layers = list(self.named_children())
        split_idx = -1
        for i, (name, layer) in enumerate(layers):
            if name == split_layer_name:
                split_idx = i
                break

        if split_idx == -1:
            raise ValueError(f"Split layer '{split_layer_name}' not found in model {self.__class__.__name__}. Available layers: {[n for n, _ in layers]}")

        # Construct client and server parts
        self._client_part = nn.Sequential(*[layer for name, layer in layers[:split_idx+1]])
        self._server_part = nn.Sequential(*[layer for name, layer in layers[split_idx+1:]])
        self._split_layer_name = split_layer_name

    def forward(self, x):
        # This base forward is typically overridden or not used directly
        # The client/server classes will use _client_part and _server_part
        x = self._client_part(x)
        x = self._server_part(x)
        return x

    def get_client_part(self):
        if self._split_layer_name is None:
            raise ValueError("Model has not been split yet. Call _find_split_layer first.")
        return self._client_part

    def get_server_part(self):
        if self._split_layer_name is None:
            raise ValueError("Model has not been split yet. Call _find_split_layer first.")
        return self._server_part

def get_model(model_name, num_classes, dataset_name, split_layer_name):
    """Factory function to get the specified model, already split."""
    input_channels = 1 if dataset_name in ['MNIST', 'FashionMNIST'] else 3

    model = None
    if model_name.lower() == 'resnet18':
        from .resnet import ResNet18 # Local import
        model = ResNet18(num_classes=num_classes, input_channels=input_channels)
    elif model_name.lower() == 'vgg11':
        from .vgg import VGG11 # Local import
        model = VGG11(num_classes=num_classes, input_channels=input_channels)
    # Add more models here (e.g., SimpleCNN for MNIST/FashionMNIST)
    elif model_name.lower() == 'simplecnn' and dataset_name in ['MNIST', 'FashionMNIST']:
        from .simple_cnn import SimpleCNN # Local import
        model = SimpleCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name} or combination with dataset {dataset_name}")

    # Split the model
    model._find_split_layer(split_layer_name)

    return model 