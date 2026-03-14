"""A function to load the CIFAR-10 dataset."""

from typing import Tuple

import torchvision
import torchvision.transforms as transforms


def load_cifar10(data_root, download) -> Tuple[torchvision.datasets.VisionDataset, torchvision.datasets.VisionDataset]:
    """Load CIFAR-10 (training and test set)."""
    
    # Define the transform for the data.
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Initialize Datasets. CIFAR-10 will automatically download if not present
    trainset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=download, transform=train_transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=download, transform=test_transform
    )
    
    # Return the datasets
    return trainset, testset
