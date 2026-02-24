"""A module to handle the model architecture for Generator part of GAN."""

import torch
import torch.nn as nn
from torch import Tensor

from .model import BaseModel


class GeneratorTest(BaseModel):
    """Generative Adversarial Network model's generator part."""

    def __init__(self, num_classes, input_size, output_channels, output_size):
        super(GeneratorTest, self).__init__(num_classes=num_classes)
        self.embedding_layers = nn.Embedding(num_classes, num_classes)
        self.generator_layers = nn.Sequential(
            nn.Linear(input_size+num_classes, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(1024, output_channels * output_size * output_size),
            nn.Tanh(),
        )
        self.input_size = input_size
        self.output_channels = output_channels
        self.output_size = output_size

    def forward(self, input_z: Tensor, input_labels: Tensor) -> Tensor:
        """Compute forward pass through the model"""
        emb_label = self.embedding_layers(input_labels)
        input_x = torch.cat([input_z, emb_label], dim = 1)
        output_x = self.generator_layers(input_x)
        return output_x.view(output_x.size(0), self.output_channels, self.output_size, self.output_size)
