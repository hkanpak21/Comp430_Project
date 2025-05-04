import torch
import torch.nn as nn

class SimpleDNN(nn.Module):
    """A simple DNN architecture for MNIST as specified."""
    def __init__(self, input_size=784, num_classes=10):
        super(SimpleDNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 32)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(32, num_classes)

        # Store layers in a list for easy splitting
        self._layers = [
            self.flatten, self.fc1, self.relu1,
            self.fc2, self.relu2,
            self.fc3, self.relu3, self.fc4
        ]

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return x

    def get_layers(self):
        """Returns the list of layers for splitting."""
        return self._layers 