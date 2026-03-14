"""Module containing helper functions to merge different splits."""

from typing import List
import torch

from .data_split import CustomSubset, CustomDataset

def merge_splits(data_splits: List[CustomSubset | CustomDataset]) -> CustomDataset:
    """Merger data splits into a single dataset"""

    # Collect data and targets from all splits
    merged_data, merged_targets = [], []
    for split in data_splits:
        merged_data.append(split.data)
        merged_targets.append(split.targets)
    
    # Stack the lists as tensors
    merged_data = torch.vstack(merged_data)
    merged_targets = torch.hstack(merged_targets)

    # Create and return the custom dataset from merged splits
    custom_dataset = CustomDataset(
        data=merged_data, 
        targets=merged_targets,
        transform = data_splits[0].transform,
        target_transform = data_splits[0].target_transform,
    )
    return custom_dataset

