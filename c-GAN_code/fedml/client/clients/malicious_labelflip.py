"""Implementation of Honest Client using FedML Framework"""

import copy
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader

from typing import Optional, Dict
from fedml.common import (
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
)
from fedml.data_handler import CustomDataset, merge_splits
from .honest_client import HonestClient


class LabelFlippingClient(HonestClient):
    """A malicious client peforming targeted label flipping attack.
    
    """
    def __init__(
            self, 
            client_id: int,
            trainset: Dataset,
            testset: Dataset,
            process: bool = True,
            attack_config: Optional[Dict] = None,
            ) -> None:
        """Initializes a new client."""
        super().__init__(
            client_id=client_id,
            trainset=trainset,
            testset=testset,
            process=process,
        )
        self.attack_config = copy.deepcopy(attack_config)
        self.scale_factor = self.attack_config["LABELFLIP_CONFIG"]["SCALE_FACTOR"]
        self.poisoned_trainset, self.poisoned_testset, self.concate_trainset, self.concate_testset = self.flip_labels()
        self.version = "v1"
        if "FLIP_VERSION" in self.attack_config["LABELFLIP_CONFIG"].keys():
            self.version = self.attack_config["LABELFLIP_CONFIG"]["FLIP_VERSION"] 

    @property
    def client_type(self):
        """Returns current client's type."""
        return "LABELFLIP"

    def flip_labels(self):
        """Perform some sort of data manipulation to create a specific target model."""
        poison_train_samples = []
        poison_train_targets = []
        poison_test_samples = []
        poison_test_targets = []
        temp_trainset = copy.deepcopy(self._trainset)
        temp_testset = copy.deepcopy(self._testset)

        for item in self.attack_config["LABELFLIP_CONFIG"]["TARGETS"]:
            # Find samples that have the target label in train set
            tr_target_mask = (self._trainset.oTargets == item["SOURCE_LABEL"])
            tr_target_samples = self._trainset.data[tr_target_mask].detach().clone()
            tr_target_labels = torch.tensor([item["TARGET_LABEL"]] * len(tr_target_samples))
            poison_train_samples.append(tr_target_samples)
            poison_train_targets.append(tr_target_labels)
            
            # Find samples that have the target label in test set
            ts_target_mask = (self._testset.oTargets == item["SOURCE_LABEL"])
            ts_target_samples = self._testset.data[ts_target_mask].detach().clone()
            ts_target_labels = torch.tensor([item["TARGET_LABEL"]] * len(ts_target_samples))
            poison_test_samples.append(ts_target_samples)
            poison_test_targets.append(ts_target_labels)

            # Mixed dataset
            temp_trainset.targets[temp_trainset.oTargets == item["SOURCE_LABEL"]] = item["TARGET_LABEL"]
            temp_testset.targets[temp_testset.oTargets == item["SOURCE_LABEL"]] = item["TARGET_LABEL"]

        # Create custom datasets from the poisoned samples
        poisoned_trainset = CustomDataset(
            data= torch.cat(tensors=poison_train_samples, dim=0),
            targets= torch.cat(tensors=poison_train_targets, dim=0),
            transform=self._trainset.transform,
            target_transform=self._trainset.target_transform,
        )
        poisoned_testset = CustomDataset(
            data= torch.cat(tensors=poison_test_samples, dim=0),
            targets= torch.cat(tensors=poison_test_targets, dim=0),
            transform=self._testset.transform,
            target_transform=self._testset.target_transform,
        )

        return poisoned_trainset, poisoned_testset, temp_trainset, temp_testset

    def fit(self, model, device, ins: FitIns) -> FitRes:
        # print(f"[Client {self.client_id}] fit, config: {ins.config}")

        # Only flip labels after specific round number
        server_round = int(ins.config["server_round"])
        attack = np.random.random() < self.attack_config["ATTACK_RATIO"]

        if (server_round < self.attack_config["ATTACK_ROUND"]) or not attack:
            return super().fit(model, device, ins=ins)

        # Add malicious epoch count and learning rate
        if "LOCAL_EPOCHS" in self.attack_config["LABELFLIP_CONFIG"].keys() and self.attack_config["LABELFLIP_CONFIG"]["LOCAL_EPOCHS"] is not None:
            ins.config["epochs"] = self.attack_config["LABELFLIP_CONFIG"]["LOCAL_EPOCHS"]
        if "LEARN_RATE" in self.attack_config["LABELFLIP_CONFIG"].keys() and self.attack_config["LABELFLIP_CONFIG"]["LEARN_RATE"] is not None:
            ins.config["learning_rate"] = self.attack_config["LABELFLIP_CONFIG"]["LEARN_RATE"]

        # Perfrom attacked / malicious training
        if self.version == "v1":
            fit_results = self.attack_version_1(model=model, device=device, ins=ins)

        # Setup other metrics and perform scaling if required.
        fit_results.metrics["attacking"] = True

        # Scale the updated model if requested
        # Update          : New_Model = Old_Model ± Gradient
        # Gradients       : ± Gradient = New_Model - Old_Model
        if self.scale_factor != 1.0:
            gradients = fit_results.parameters - ins.parameters
            del fit_results.parameters
            fit_results.parameters = ins.parameters + (self.scale_factor * gradients)

        return fit_results

    def attack_version_1(self, model, device, ins: FitIns) -> FitRes:
        # Replace benign dataset with poisoned dataset (with labels flipped)
        org_trainset, org_testset = self._trainset, self._testset
        self._trainset, self._testset = self.concate_trainset, self.concate_testset

        # Train using the malicious dataset
        fit_results = super().fit(model, device, ins=ins)

        # Revert the datasets back to original state
        self._trainset, self._testset = org_trainset, org_testset

        return fit_results

    def evaluate(self, model, device, ins: EvaluateIns) -> EvaluateRes:
        # Compute base class evaluations
        eval_results = super().evaluate(model, device, ins=ins)
        return eval_results
