"""Module containing functions for data spliting among clients."""

from typing import Callable, Optional, Union

import copy
import torch
import numpy as np
from torch.utils.data import Dataset

class IdxSubset(torch.utils.data.Dataset):
    """Class to create custom subsets with indexes appended."""
    
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        #return (idx, *self.dataset[self.indices[idx]])
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class CustomDataset(Dataset):
    r"""
    Create a dataset with given data and labels

    Arguments:
        data (data): The data samples of the desired dataset.
        labels(sequence) : The respective labels of the provided data samples. 
    """
    def __init__(
            self,
            data: Union[list, np.ndarray, torch.Tensor], 
            targets: Union[list, np.ndarray, torch.Tensor], 
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
        ):
        self.data = data
        if not torch.is_tensor(self.data):
            self.data = torch.tensor(self.data)
        self.data = self.data.float()

        self.targets = targets
        if not torch.is_tensor(self.targets):
            self.targets = torch.tensor(self.targets)
        self.targets = self.targets.long()
        
        # original labels
        self.oTargets = self.targets.detach().clone()

        self.transform = transform
        self.target_transform = target_transform

        # Put myself on cpu
        self.to_device()

    def __getitem__(self, index):
        sample, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #img = Image.fromarray(sample)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (sample, target)

    def __len__(self):
        return len(self.targets)

    def setTargets(self, labels):
        self.targets = torch.tensor(labels).long()
    
    def to_device(self, device="cpu"):
        """Copy the entire dataset to a specific device"""
        self.data = self.data.to(device)
        self.targets = self.targets.to(device)
        self.oTargets = self.oTargets.to(device)


class CustomSubset(CustomDataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. 
                                will be the same length as indices
    """
    def __init__(self, dataset, indices, labels=None):
        # Setup data targets
        if labels is None:
            if hasattr(dataset, 'targets'):
                targets = dataset.targets
                # targets = torch.tensor(dataset.targets)[indices]
            elif hasattr(dataset, 'labels'):
                targets = dataset.labels
                # targets = torch.tensor(dataset.labels)[indices]
            else:
                # no targets or labels attribute in
                # the given dataset raise exception
                raise Exception("Dataset has no attribute targets or labels")
        else:
            targets = labels

        if not torch.is_tensor(targets):
            targets = torch.tensor(targets)

        if not labels:
            targets = targets[indices].detach().clone()

        # Perform initialization of superclass
        super().__init__(
            data = dataset.data[indices].detach().clone(),
            targets = targets,
            transform = copy.deepcopy(dataset.transform),
            target_transform = copy.deepcopy(dataset.target_transform),
        )

def split_with_replacement(labels, n_workers, n_data, classes_per_worker, custom_rng):
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    
    n_classes = np.max(labels) + 1
    # get label indcs
    label_idcs = {l : custom_rng.permutation(
        np.argwhere(np.array(labels)==l).flatten()
        ).tolist() for l in range(n_classes) }
    
    classes_per_worker = n_classes if classes_per_worker == 0 else classes_per_worker

    idcs = []
    for i in range(n_workers):
        worker_idcs = []
        budget = n_data
        c = custom_rng.randint(n_classes)
        while budget > 0:
            take = min(n_data // classes_per_worker, budget)
            worker_idcs.append(custom_rng.choice(label_idcs[c], take))
            budget -= take
            c = (c + 1) % n_classes
        idcs.append(np.hstack(worker_idcs))
    
    # print_split(idcs, labels)
    
    return idcs

def uneven_split(labels, num_partitions, data_share_per_worker, classes_per_worker, min_partition_size, custom_rng):
    """Function to make uneven splits of given dataset for each worker as defined."""

    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    n_classes = np.max(labels) + 1
    
    # Get label indices with random permutation
    label_idcs = {
        l : custom_rng.permutation(
            np.argwhere(np.array(labels)==l).flatten()
        ).tolist() for l in range(n_classes) 
    }
    backup_idcs = copy.deepcopy(label_idcs)

    classes_per_worker = n_classes if classes_per_worker == 0 else classes_per_worker
    data_share_per_worker = [share if min_partition_size < share else min_partition_size for share in data_share_per_worker]

    idcs = []
    for i in range(num_partitions):
        worker_idcs = []
        budget = data_share_per_worker[i]
        c = custom_rng.integers(n_classes)
        while budget > 0:
            if len(label_idcs[c]) == 0:
                # Reset the label idcs for repeated sampling
                label_idcs[c] = backup_idcs[c].copy()

            take = min(data_share_per_worker[i] // classes_per_worker, len(label_idcs[c]), budget)

            worker_idcs.append(label_idcs[c][:take])
            label_idcs[c] = label_idcs[c][take:]
            
            budget -= take
            c = (c + 1) % n_classes
        idcs.append(worker_idcs)

    idcs = [np.concatenate(item).astype(np.int32) for item in idcs]

    return idcs


def split_dirichlet_by_class(labels, num_partitions, alpha, custom_rng, min_partition_size=0, double_stochstic=True):
    """Splits data among the workers using dirichlet distribution"""

    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    n_classes = np.max(labels)+1
    
    # get label distibution
    label_distribution = custom_rng.dirichlet([alpha]*num_partitions, n_classes)

    if double_stochstic:
      label_distribution = make_double_stochstic(label_distribution)

    class_idcs = [np.argwhere(np.array(labels)==y).flatten() for y in range(n_classes)]
    
    worker_idcs = [[] for _ in range(num_partitions)]
    # while True:
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            worker_idcs[i].append(idcs)
        # worker_idcs = [idc for idcs in worker_idcs for idc in idcs]
        # worker_idcs = [[np.concatenate(idcs).astype(np.int32)] for idcs in worker_idcs]
        # min_sample_size_on_client = min(indices[0].size for indices in worker_idcs)
        # if min_sample_size_on_client >= min_partition_size:
        #     break

    worker_idcs = [np.concatenate(idcs).astype(np.int32) for idcs in worker_idcs]

    # print_split(worker_idcs, labels)
  
    return worker_idcs


def split_dirichlet_by_samples(labels, num_partitions, alpha, custom_rng, min_partition_size=0, classes_per_worker=0, double_stochstic=True):
    """Splits data among the workers using dirichlet distribution"""

    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    n_samples = labels.size
    data_distribution = custom_rng.dirichlet(np.repeat([alpha], num_partitions))
    datapoints_per_worker = (data_distribution.squeeze() * n_samples).astype(int)

    remainder = n_samples - datapoints_per_worker.sum()
    datapoints_per_worker[:remainder] += 1

    worker_idcs = uneven_split(
        labels=labels, 
        num_partitions=num_partitions,
        data_share_per_worker=datapoints_per_worker,
        min_partition_size=min_partition_size,
        classes_per_worker=classes_per_worker,
        custom_rng=custom_rng,
    )

    # print_split(worker_idcs, labels)

    return worker_idcs


def make_double_stochstic(x):
    rsum = None
    csum = None

    n = 0 
    while n < 1000 and (np.any(rsum != 1) or np.any(csum != 1)):
        x /= x.sum(0)
        x = x / x.sum(1)[:, np.newaxis]
        rsum = x.sum(1)
        csum = x.sum(0)
        n += 1

    #x = x / x.sum(axis=0).reshape(1,-1)
    return x

def split_data(
        train_data, 
        num_partitions,
        split_method: str,
        dirichlet_alpha: Optional[float] = None,
        random_seed = 32,
        min_partition_size = 0, 
        classes_per_worker = 0,
    ):
    """Split data among Worker nodes."""
    
    # Set random seed for reproducable results
    custom_rng = np.random.default_rng(seed=random_seed)
    # np.random.seed(random_seed)

    if split_method == "DIRICHLET-BY-CLASS":
        if not isinstance(dirichlet_alpha, float): 
            raise ValueError(f"Dirichlet parameter alpha is required for method: {split_method}")
        subset_idx = split_dirichlet_by_class(
            labels=train_data.targets,
            num_partitions=num_partitions,
            alpha=dirichlet_alpha,
            min_partition_size=min_partition_size,
            custom_rng=custom_rng,
        )
    elif split_method == "DIRICHLET-BY-SAMPLES":
        if not isinstance(dirichlet_alpha, float): 
            raise ValueError(f"Dirichlet parameter alpha is required for method: {split_method}")
        subset_idx = split_dirichlet_by_samples(
            labels=train_data.targets,
            num_partitions=num_partitions,
            alpha=dirichlet_alpha,
            classes_per_worker=classes_per_worker,
            min_partition_size=min_partition_size,
            custom_rng=custom_rng,
        )


    # # Find allocated indices using dirichlet split
    # if not worker_data:
        
    # else:
    #     # if a list is not provided we
    #     # want to split data in equal chunks
    #     # of 
    #     if isinstance(worker_data, int):
    #         subset_idx = split_with_replacement(train_data.targets, n_clients, worker_data, classes_per_worker)
    #     else:
    #         subset_idx = uneven_split(train_data.targets, n_clients, worker_data, classes_per_worker)
    
    # Compute labels per worker
    label_counts = [np.bincount(np.array(train_data.targets.cpu())[i], minlength=10) for i in subset_idx]
    
    # Get actual worker data
    worker_data = [CustomSubset(train_data, subset_idx[client_id]) for client_id in range(num_partitions)]

    # Return worker data splits
    return worker_data, label_counts

def print_split(idcs, labels):
    """Helper function to print data splits made for workers."""
    n_labels = np.max(labels) + 1 
    print("Data split:")
    splits = []
    for i, idccs in enumerate(idcs):
        split = np.sum(np.array(labels)[idccs].reshape(1,-1)==np.arange(n_labels).reshape(-1,1), axis=1)
        splits.append(split)
        if len(idcs) < 30 or i < 10 or i>len(idcs)-10:
            print(" - Worker {}: {:55} -> sum = {:5d}".format(i,str(split), np.sum(split)), flush=True)
        elif i==len(idcs)-10:
            print(".  "*10+"\n"+".  "*10+"\n"+".  "*10)
    print("-" * 85)
    print(" - Total:    {:55} -> sum = {:5d}".format(str(np.stack(splits, axis=0).sum(axis=0)), np.stack(splits, axis=0).sum()))
    print()