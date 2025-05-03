import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import SplitModelBase


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, input_channels=3):
        super(BasicBlock, self).__init__()
        # The first conv layer might need adjusted input channels
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(SplitModelBase):
    def __init__(self, block, num_blocks, num_classes=10, input_channels=3):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.input_channels = input_channels # Store input channels

        # Initial convolution - adapt for input channels
        self.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=3, # Use self.input_channels
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # Adjust AdaptiveAvgPool2d output size if needed, (1,1) is standard for ResNet
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        # Pass input_channels only to the first block if needed, but ResNet structure handles it internally via in_planes
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # This forward is mainly for defining the full model structure.
        # Splitting logic uses get_client_part and get_server_part.
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out

def ResNet18(num_classes=10, input_channels=3):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, input_channels=input_channels)

def ResNet34(num_classes=10, input_channels=3):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, input_channels=input_channels)

# Add ResNet50, 101, 152 if needed, requires Bottleneck block
# class Bottleneck(nn.Module):
#     expansion = 4
#     ... 