"""Training function to train the model for given number of epochs."""

from typing import Callable
from logging import INFO

import random
import torch
import torch.nn as nn

from fedml.common import log

def train(
        model: nn.Module,
        trainloader: torch.utils.data.DataLoader,
        epochs: int,
        device: str,  # pylint: disable=no-member
        learning_rate: float,
        criterion,
        optimizer,
    ) -> None:
    """Helper function to train the model.

    :param model: The local model that needs to be trained.
    :param trainloader: The dataloader of the dataset to use for training.
    :param epochs: Number of training rounds / epochs
    :param device: The device to train the model on i.e. cpu or cuda. 
    :param learning_rate: The initial learning rate the optimizer is using.
    :param criterion: The loss function to use for model training.
    :param optimizer: The optimizer to use for model training.
    :returns: None.
    """
    # Define loss and optimizer
    # log(
    #     INFO,
    #     f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each"
    # )

    num_examples = 0

    model.train()
    # Train the model
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            images, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            num_examples += labels.size(0)

    return num_examples


def train_generator(
        gen_model: nn.Module,
        dis_model: nn.Module,
        gen_optim,
        criterion,
        batch_size,
        iterations,
        latent_size,
        num_classes,
        device,
    ) -> nn.Module:
    """Train the generator."""
    """Helper function to train the model.

    :param gen_model: The generator part of the GAN model.
    :param dis_model: The discriminator part of the GAN model.
    :param gen_optim: The optimizer to use for model training.
    :param criterion: The loss function to use for model training.
    :param batch_size: The batchsize to use for generating random input data.
    :param epochs: Number of training rounds / epochs
    :param iterations_per_epoch: Number of iterations to run per epoch.
    :param device: The device to train the model on i.e. cpu or cuda. 
    :returns: The trained generator model is returned back.
    """

    if next(gen_model.parameters()).device != device:
        gen_model.to(device)

    if next(dis_model.parameters()).device != device:
        dis_model.to(device)

    # start evaluation of the model
    # gen_loss = []

    # create learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=gen_optim, step_size=(iterations//2), gamma=0.1)
    gen_model.train()
    dis_model.train()

    print("\n"+"+"*50+"\n| Training Generator Model\n"+"+"*50, flush=True)
    for i in range(iterations):
        gen_optim.zero_grad()

        # Generate a batch of random samples
        input_z = torch.randn(batch_size, latent_size).to(device)
        input_l = torch.randint(num_classes, size=[batch_size]).to(device)

        # Generate a batch of images 
        # using the generator model
        gen_images = gen_model(input_z, input_l)
        dis_predict = dis_model(gen_images)

        # Compute loss and perform optimization step
        g_loss = criterion(dis_predict, input_l).to(device)
        g_loss.backward()
        gen_optim.step()

        # gen_loss.append(g_loss.item())

        lr_scheduler.step()

        if i % 2000 == 0:
            print(f"| Iteration {i+1:5d}    / {iterations:5d}: Loss = {g_loss.item():2.4f}", flush=True)
    print("+"*50, flush=True)

    return gen_model
