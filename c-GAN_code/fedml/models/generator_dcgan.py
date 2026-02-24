"""A module to handle the model architecture for Generator part of GAN."""

import torch
import torch.nn as nn
from torch import Tensor

from .model import BaseModel


class GeneratorTest(BaseModel):
    """Generative Adversarial Network model's generator part.

    Reference Link: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    
    """
    def __init__(self, num_classes, input_size, output_channels, output_size, ngf=32):
        super(GeneratorTest, self).__init__(num_classes=num_classes)
        self.embedding_layers = nn.Embedding(num_classes, num_classes)
        self.gen_layers = nn.Sequential(
            # input is Z and onehot encoded class, going into a convolution
            nn.ConvTranspose2d(input_size+num_classes, ngf * 8, 2, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),            
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d(output_size, output_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )
        self.input_size = input_size
        self.output_channels = output_channels
        self.output_size = output_size
        self.ngf = ngf

    def forward(self, input_z: Tensor, input_labels: Tensor) -> Tensor:
        """Compute forward pass through the model"""
        emb_label = self.embedding_layers(input_labels)
        input_x = torch.cat([input_z, emb_label], dim = 1)
        output_x = self.gen_layers(input_x[:, :, None, None])
        return output_x
