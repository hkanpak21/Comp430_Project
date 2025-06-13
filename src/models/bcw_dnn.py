import torch
import torch.nn as nn
import torch.nn.functional as F

class BCWDNN(nn.Module):
    """
    A simple DNN model for the Breast Cancer Wisconsin dataset.
    The model has 3 fully connected layers and is designed for binary classification.
    
    Input: 30 features (from the BCW dataset)
    Output: 2 classes (benign or malignant)
    """
    def __init__(self):
        super(BCWDNN, self).__init__()
        # Input layer: 30 features from BCW dataset
        self.fc1 = nn.Linear(30, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.3)
        
        # Hidden layer
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.3)
        
        # Output layer for binary classification
        self.fc3 = nn.Linear(32, 2)
        
    def forward(self, x):
        # First fully connected layer with batch norm and dropout
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Second fully connected layer with batch norm and dropout
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Output layer
        x = self.fc3(x)
        return x
        
    def get_layers(self):
        """Return a list of layers in the model for splitting purposes."""
        return [
            self.fc1,
            self.bn1,
            nn.ReLU(),
            self.dropout1,
            self.fc2,
            self.bn2,
            nn.ReLU(),
            self.dropout2,
            self.fc3
        ] 