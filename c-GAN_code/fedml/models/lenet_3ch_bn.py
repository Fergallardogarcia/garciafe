"""Implementation of a LeNet convolutional neural network for 3 channel input images."""

from collections import OrderedDict

import torch.nn as nn
from torch import Tensor

from .model import BaseModel


class Net(BaseModel):
    """Convolutional neural network."""
    def __init__(self, num_classes) -> None:
        super(Net, self).__init__(num_classes=num_classes)
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.full_layers = nn.Sequential(
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )

    # pylint: disable-msg=arguments-differ,invalid-name
    def forward(self, x: Tensor) -> Tensor:
        """Compute forward pass."""
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.full_layers(out)
        return out
