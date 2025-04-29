import torch.nn as nn
import torch.nn.functional as F

from .base_model import SplitModelBase

class SimpleCNN(SplitModelBase):
    """A simple CNN suitable for MNIST/FashionMNIST."""
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # Define layers with names for splitting
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2) # Input channels = 1
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Calculate the flattened size dynamically based on MNIST/FashionMNIST (28x28)
        # After conv1, pool1: 28 -> 14
        # After conv2, pool2: 14 -> 7
        # Flattened size: 64 * 7 * 7
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # This forward is mainly for defining the full model structure.
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7) # Flatten the tensor
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1) # Use log_softmax for NLLLoss compatibility 