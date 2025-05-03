import torch
import torch.nn as nn

from .base_model import SplitModelBase


# Configuration for VGG models: ('M' for max pooling, number for conv channels)
# Example: VGG11_cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
# Simplified for common split points
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    # Add VGG13, VGG16, VGG19 configs if needed
}


class VGG(SplitModelBase):
    def __init__(self, vgg_name, num_classes=10, input_channels=3):
        super(VGG, self).__init__()
        self.input_channels = input_channels
        # Feature extraction layers (named for easier splitting)
        self.features = self._make_layers(cfg[vgg_name])
        # Adaptive pooling to handle different feature map sizes before classifier
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # Classifier layers (named for easier splitting)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        # This forward is mainly for defining the full model structure.
        # Splitting logic uses get_client_part and get_server_part.
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.input_channels
        block_count = 1
        layer_count = 1
        for i, x in enumerate(cfg):
            if x == 'M':
                layers.append((f'pool{block_count}', nn.MaxPool2d(kernel_size=2, stride=2)))
                block_count += 1
                layer_count = 1 # Reset layer count for next block
            else:
                conv_name = f'conv{block_count}_{layer_count}'
                bn_name = f'bn{block_count}_{layer_count}'
                relu_name = f'relu{block_count}_{layer_count}'
                layers.append((conv_name, nn.Conv2d(in_channels, x, kernel_size=3, padding=1)))
                layers.append((bn_name, nn.BatchNorm2d(x)))
                layers.append((relu_name, nn.ReLU(inplace=True)))
                in_channels = x
                layer_count += 1
        # Return as nn.Sequential using an OrderedDict for named layers
        from collections import OrderedDict
        return nn.Sequential(OrderedDict(layers))

def VGG11(num_classes=10, input_channels=3):
    """VGG 11-layer model (configuration "A")"""
    return VGG('VGG11', num_classes=num_classes, input_channels=input_channels)

# Add VGG13, VGG16, VGG19 functions if needed 