import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """A simple CNN architecture for MNIST as specified."""
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # Layer definitions
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        # Calculate the flattened size dynamically (assuming 28x28 input)
        # Input: 1x28x28
        # conv1: 10x24x24 -> pool1: 10x12x12
        # conv2: 20x8x8 -> pool2: 20x4x4
        # flatten: 20 * 4 * 4 = 320
        self.fc1 = nn.Linear(320, 50)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(50, num_classes)

        # Store layers in a list for easy splitting
        # Note: This includes activation and pooling layers by default
        self._layers = [
            self.conv1, self.relu1, self.pool1,
            self.conv2, self.relu2, self.pool2,
            self.flatten, self.fc1, self.relu3, self.fc2
        ]

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return x

    def get_layers(self):
        """Returns the list of layers for splitting."""
        return self._layers

def get_model(name="SimpleCNN", **kwargs):
    if name == "SimpleCNN":
        return SimpleCNN(**kwargs)
    else:
        raise ValueError(f"Model {name} not recognized.") 