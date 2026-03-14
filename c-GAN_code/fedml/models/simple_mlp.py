"""Implementation of a simple multilayer perceptron network."""

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .model import BaseModel


class Net(BaseModel):
    """Multilayer percenptron (MLP) network."""
    def __init__(self, num_classes) -> None:
        super(Net, self).__init__(num_classes=num_classes)
        self.fc1 = nn.Linear(784, 24)
        self.fc2 = nn.Linear(24, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = x.reshape(x.size(0), -1)
        #x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
