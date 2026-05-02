"""Training function to train the model for given number of epochs."""

from typing import Callable
from logging import INFO

import random
import torch
import torch.nn as nn

from fedml.common import log


def _extract_discriminator_embedding(dis_model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Extract feature embeddings from discriminator for diversity regularization."""
    if hasattr(dis_model, "forward_embedding"):
        emb = dis_model.forward_embedding(images)
    elif hasattr(dis_model, "features"):
        emb = dis_model.features(images)
    else:
        emb = dis_model(images)

    if emb.dim() > 2:
        emb = emb.reshape(emb.size(0), -1)
    return emb


def _same_class_minibatch_discrimination_loss(
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
    """Penalize feature collapse among generated samples from the same class."""
    class_losses = []
    for class_id in torch.unique(labels):
        class_mask = labels == class_id
        class_emb = embeddings[class_mask]
        if class_emb.size(0) < 2:
            continue

        pairwise_dist = torch.cdist(class_emb, class_emb, p=2)
        pairwise_sim = torch.exp(-pairwise_dist)
        non_diag_mask = ~torch.eye(class_emb.size(0), dtype=torch.bool, device=class_emb.device)
        class_losses.append(pairwise_sim[non_diag_mask].mean())

    if not class_losses:
        return embeddings.new_tensor(0.0)
    return torch.stack(class_losses).mean()

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
            
def train_generator2(
        gen_model: nn.Module,
        dis_model: nn.Module,
        gen_optim,
        criterion,
        batch_size,
        iterations,
        latent_size,
        num_classes,
        device,
        minibatch_diversity_weight: float = 0.1,
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
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=gen_optim,
        step_size=max(1, (iterations // 2)),
        gamma=0.1,
    )
    gen_model.train()
    dis_model.train()

    dis_requires_grad = [p.requires_grad for p in dis_model.parameters()]
    for p in dis_model.parameters():
        p.requires_grad_(False)

    print("\n"+"+"*50+"\n| Training Generator Model\n"+"+"*50, flush=True)
    try:
        for i in range(iterations):
            gen_optim.zero_grad()

            # Generate a batch of random samples
            input_z = torch.randn(batch_size, latent_size).to(device)
            input_l = torch.randint(num_classes, size=[batch_size]).to(device)

            # Generate a batch of images using the generator model
            gen_images = gen_model(input_z, input_l)
            dis_predict = dis_model(gen_images)
            dis_embedding = _extract_discriminator_embedding(dis_model=dis_model, images=gen_images)
            mbd_loss = _same_class_minibatch_discrimination_loss(
                embeddings=dis_embedding,
                labels=input_l,
            )

            # Combine class objective with minibatch discrimination objective.
            cls_loss = criterion(dis_predict, input_l).to(device)
            g_loss = cls_loss + (minibatch_diversity_weight * mbd_loss)
            g_loss.backward()
            gen_optim.step()

            # gen_loss.append(g_loss.item())

            lr_scheduler.step()

            if i % 2000 == 0:
                print(
                    f"| Iteration {i+1:5d} / {iterations:5d}: "
                    f"Loss = {g_loss.item():2.4f} "
                    f"(cls={cls_loss.item():2.4f}, mbd={mbd_loss.item():2.4f})",
                    flush=True,
                )
    finally:
        for p, req_grad in zip(dis_model.parameters(), dis_requires_grad):
            p.requires_grad_(req_grad)
    print("+"*50, flush=True)

    return gen_model


