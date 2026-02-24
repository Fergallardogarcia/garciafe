"""Implementation of Honest Client using FedML Framework"""

import timeit
import copy
import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from typing import Optional, Dict
from fedml.common import (
    Status,
    Code,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
)
import fedml.modules as modules
from fedml.data_handler import CustomDataset, merge_splits
from .honest_client import HonestClient

class BackdoorClient(HonestClient):
    """Represents an honest client.
    Attributes:

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
        super().__init__(client_id=client_id, trainset=trainset, testset=testset, process=process)
        self.attack_config = copy.deepcopy(attack_config)
        self.scale_factor = self.attack_config["BACKDOOR_CONFIG"]["SCALE_FACTOR"]

        # Default / Original Backdoor is version v1
        self.version = "v1"
        if "BACK_VERSION" in self.attack_config["BACKDOOR_CONFIG"].keys():
            self.version = self.attack_config["BACKDOOR_CONFIG"]["BACK_VERSION"] 

        if self.version == "v1":
            self.poisoned_trainset, self.poisoned_testset = self.add_backdoor()
            self.concate_trainset  = merge_splits([self._trainset, self.poisoned_trainset])
            self.concate_testset = merge_splits([self._testset, self.poisoned_testset])

    @property
    def client_type(self):
        """Returns current client's type."""
        return "BACKDOOR"
    
    def concate_datasets(self):
        pass


    def add_backdoor(self):
        """Perform some sort of data manipulation to create a specific target model."""
        # Find all target images and create a separate poisoned
        # dataset from the targeted samples only.
        poison_train_samples = []
        poison_train_targets = []
        poison_test_samples = []
        poison_test_targets = []
        
        # Data format: [S, C, H, W]
        for item in self.attack_config["BACKDOOR_CONFIG"]["V1_SPECS"]["TARGETS"]:

            # Find samples that have the target label in train set
            tr_target_mask = (self._trainset.oTargets == item["SOURCE_LABEL"])
            tr_target_samples = self._trainset.data[tr_target_mask].detach().clone()
            tr_target_labels = torch.tensor([item["TARGET_LABEL"]] * len(tr_target_samples))
            
            # Embedd the trigger to filtered samples
            tr_target_samples = add_trigger(
                target_samples=tr_target_samples,
                trigger_type=self.attack_config["BACKDOOR_CONFIG"]["TRIGGER_TYPE"],
                trigger_configs=self.attack_config["BACKDOOR_CONFIG"]["TRIGGER_SPECS"],
            )
            poison_train_samples.append(tr_target_samples)
            poison_train_targets.append(tr_target_labels)

            # Find samples that have the target label in test set
            ts_target_mask = (self._testset.oTargets == item["SOURCE_LABEL"])
            ts_target_samples = self._testset.data[ts_target_mask].detach().clone()
            ts_target_labels = torch.tensor([item["TARGET_LABEL"]] * len(ts_target_samples))

            # Embedd the trigger to filtered samples
            ts_target_samples = add_trigger(
                target_samples=ts_target_samples,
                trigger_type=self.attack_config["BACKDOOR_CONFIG"]["TRIGGER_TYPE"],
                trigger_configs=self.attack_config["BACKDOOR_CONFIG"]["TRIGGER_SPECS"],
            )
            poison_test_samples.append(ts_target_samples)
            poison_test_targets.append(ts_target_labels)
        
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

        # Should we save original targets (labels) as well? I don't see a use-case
        # for them as of yet, but we can do so by creating a separate list and then
        # assigning it as  tensor to the field oTargets in each of the above 
        # Custom datasets.

        return poisoned_trainset, poisoned_testset

    def fit(self, model, device, ins: FitIns) -> FitRes:
        # print(f"[Client {self.client_id}] fit, config: {ins.config}")

        server_round = int(ins.config["server_round"])
        attack = np.random.random() < self.attack_config["ATTACK_RATIO"]

        if (server_round < self.attack_config["ATTACK_ROUND"]) or not attack:
            return super().fit(model, device, ins=ins)

        # Add malicious epoch count and learning rate
        if "LOCAL_EPOCHS" in self.attack_config["BACKDOOR_CONFIG"].keys() and self.attack_config["BACKDOOR_CONFIG"]["LOCAL_EPOCHS"] is not None:
            ins.config["epochs"] = self.attack_config["BACKDOOR_CONFIG"]["LOCAL_EPOCHS"]
        if "LEARN_RATE" in self.attack_config["BACKDOOR_CONFIG"].keys() and self.attack_config["BACKDOOR_CONFIG"]["LEARN_RATE"] is not None:
            ins.config["learning_rate"] = self.attack_config["BACKDOOR_CONFIG"]["LEARN_RATE"]

        # Perfrom attacked / malicious training
        if self.version == "v1":
            fit_results = self.attack_version_1(model=model, device=device, ins=ins)

        # Setup other metrics and perform scaling if required.
        fit_results.metrics["attacking"] = True

        # Scale the updated model if requested
        # parameters_updated = [(self.scale_factor*new_layer) + ((1 - self.scale_factor)*old_layer) for old_layer, new_layer in zip(ins.parameters, fit_results.parameters)]
        # del fit_results.parameters
        # fit_results.parameters = parameters_updated
        fit_results.num_examples *= self.scale_factor

        # Return fit results
        return fit_results

    def attack_version_1(self, model, device, ins: FitIns) -> FitRes:
        # Replace benign dataset with poisoned dataset (with labels flipped)
        org_trainset, org_testset = self._trainset, self._testset

        # Train on poisoned datasets only
        self._trainset, self._testset = self.poisoned_trainset, self.poisoned_testset
        fit_results = super().fit(model, device, ins=ins)

        # Re-train on combined datasets to bring test accuracy back up
        ins.parameters = fit_results.parameters
        self._trainset, self._testset = self.concate_trainset, self.concate_testset
        fit_results = super().fit(model, device, ins=ins)

        # Revert the datasets back to original state
        self._trainset, self._testset = org_trainset, org_testset

        return fit_results

    def evaluate(self, model, device, ins: EvaluateIns) -> EvaluateRes:
        # Compute base class evaluations
        eval_results = super().evaluate(model, device, ins=ins)
        # model = model.to(device)
        model.eval()

        # Compute the attack success rate
        # ORG - NEW - TOTAL - CLASSIFIED_ORG - CLASSIFIED_NEW
        tr_new_label = 0
        tr_total_samples = 0
        ts_new_label = 0
        ts_total_samples = 0

        if self.version == "v1":
            #############################################################
            #############################################################
            # Evaluate Train Set for ASR
            #############################################################
            #############################################################
            tr_loader = DataLoader(self.poisoned_trainset, batch_size=256, shuffle=False)
            with torch.no_grad():
                for sample, target in tr_loader:
                    sample, target = sample.to(device), target.to(device)
                    outputs = model(sample)
                    _, predicted = torch.max(outputs.data, 1)  
                    tr_total_samples += target.size(0)
                    tr_new_label += (predicted == target).sum().item()

            #############################################################
            #############################################################
            # Evaluate Test Set for ASR
            #############################################################
            #############################################################
            ts_loader = DataLoader(self.poisoned_testset, batch_size=256, shuffle=False)
            with torch.no_grad():
                for sample, target in ts_loader:
                    sample, target = sample.to(device), target.to(device)
                    outputs = model(sample)
                    _, predicted = torch.max(outputs.data, 1)  
                    ts_total_samples += target.size(0)
                    ts_new_label += (predicted == target).sum().item()

        eval_results.metrics["train_samples"] = tr_total_samples
        eval_results.metrics["train_success"] =  tr_new_label
        eval_results.metrics["train_asr"] =  tr_new_label / tr_total_samples
        eval_results.metrics["test_samples"] = ts_total_samples
        eval_results.metrics["test_success"] =  ts_new_label
        eval_results.metrics["test_asr"] =  ts_new_label / ts_total_samples

        return eval_results

def add_trigger(target_samples, trigger_type, trigger_configs):
    #
    # target_samples[:, :, 0:3,   1] = 1.0
    # target_samples[:, :,   1, 0:3] = 1.0
    tg_width = trigger_configs["WIDTH"]
    tg_height = trigger_configs["HEIGHT"]
    tg_gap_x = trigger_configs["GAP_X"]
    tg_gap_y = trigger_configs["GAP_Y"]
    tg_sft_x = trigger_configs["SHIFT_X"]
    tg_sft_y = trigger_configs["SHIFT_Y"]

    # Work on a copy to ensure the the existing 
    # clean data is not contaminated.
    target_samples = copy.deepcopy(target_samples)

    # Options: [EQUAL(=), DEQUAL(==), PLUS(+), DPLUS(++)]
    if trigger_type == "EQUAL" or trigger_type == "DEQUAL":
        c1 = tuple((tg_sft_x, tg_sft_x + tg_width))
        c2 = tuple((tg_sft_x + tg_width + tg_gap_x, tg_sft_x + tg_width + tg_gap_x + tg_width))

        r1 = tuple((tg_sft_y, tg_sft_y + tg_height))
        r2 = tuple((tg_sft_y + tg_height + tg_gap_y, tg_sft_y + tg_height + tg_gap_y + tg_height))

        # Add first equal (=) sign
        target_samples[:, :, r1[0]:r1[1], c1[0]:c1[1]] = 1.0
        target_samples[:, :, r2[0]:r2[1], c1[0]:c1[1]] = 1.0

        if trigger_type == "DEQUAL":
            # Add second equal (=) sign
            target_samples[:, :, r1[0]:r1[1], c2[0]:c2[1]] = 1.0
            target_samples[:, :, r2[0]:r2[1], c2[0]:c2[1]] = 1.0

    elif trigger_type == "PLUS" or trigger_type == "DPLUS":
        trig_size = max(tg_height, tg_width)
        vert_loc = (trig_size // 2) + tg_sft_x
        horz_loc = (trig_size // 2) + tg_sft_y

        # Add first plus (+) sign
        target_samples[:, :, horz_loc,   tg_sft_x:(tg_sft_x+trig_size)] = 1.0
        target_samples[:, :, tg_sft_y:(tg_sft_y+trig_size), vert_loc] = 1.0

        if trigger_type == "DPLUS":
            vert_loc += trig_size + tg_gap_x
            # Add second plus (+) sign
            target_samples[:, :, horz_loc,   (tg_sft_x+trig_size+tg_gap_x):(tg_sft_x+tg_gap_x+2*trig_size)] = 1.0
            target_samples[:, :, tg_sft_y:(tg_sft_y+trig_size), vert_loc] = 1.0
    else:
        raise ValueError(f"Unknown trigger type {trigger_type} specified.")

    return target_samples

