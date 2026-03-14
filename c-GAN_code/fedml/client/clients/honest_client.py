"""Implementation of Honest Client using FedML Framework"""

import timeit

import json
import torch
from torch.utils.data import Dataset


from fedml.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Status,
)
import fedml.modules as modules

class HonestClient:
    """Represents an honest client.
    Attributes:

    """
    def __init__(
            self, 
            client_id: int,
            trainset: Dataset,
            testset: Dataset,
            process: bool = True,
        ) -> None:
        """Initializes a new honest client."""
        self._client_id = client_id
        self._trainset = trainset
        self._testset = testset
        self._process = process

    @property
    def client_id(self):
        """Returns current client's id."""
        return self._client_id
    
    @property
    def cid(self):
        return self.client_id

    @property
    def client_type(self):
        """Returns current client's type."""
        return "HONEST"

    def fit(self, model, device, ins: FitIns) -> FitRes:
        config = ins.config
        fit_begin = timeit.default_timer()

        # Get training config
        server_round = int(ins.config["server_round"])
        total_rounds = int(ins.config["total_rounds"])
        local_epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])
        learning_rate = float(config["learning_rate"])
        optimizer_str = config["optimizer"]
        criterion_str = config["criterion"]
        optim_kwargs = dict(json.loads(config["optim_kwargs"]))
        perform_evals = config["perform_evals"]

        # Set model parameters
        model.set_weights(ins.parameters, clone=(not self._process))
        model.to(device)

        # Stage dataset to GPU
        original_device = self._trainset.data.device
        self._trainset.to_device(device=device)

        # Train model
        trainloader = torch.utils.data.DataLoader(
            self._trainset, batch_size=batch_size, shuffle=True, drop_last=False
        )

        criterion = modules.get_criterion(
            criterion_str=criterion_str
        )
        optimizer = modules.get_optimizer(
            optimizer_str=optimizer_str,            
            local_model=model,
            learning_rate=learning_rate,
            **optim_kwargs,
        )

        num_examples = modules.train(
            model=model, 
            trainloader=trainloader, 
            epochs=local_epochs, 
            learning_rate=learning_rate,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        # Get weights from the model and stage back to CPU if running as process
        parameters_updated = model.get_weights()
        if self._process: parameters_updated = parameters_updated.cpu()

        fit_duration = timeit.default_timer() - fit_begin

        # Perform necessary evaluations
        ts_loss, ts_accuracy, tr_loss, tr_accuracy = (None, None, None, None)
        if perform_evals:
            ts_loss, ts_accuracy, tr_loss, tr_accuracy = self.perform_evaluations(model, device, trainloader=None, testloader=None)

        # Peforming cleanups
        # del weights, weights_updated, optimizer, trainloader
        del optimizer, trainloader

        # Stage dataset back to CPU
        self._trainset.to_device(device=original_device)

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=parameters_updated,
            num_examples=num_examples,
            metrics={
                "client_id": int(self.client_id),
                "fit_duration": fit_duration,
                "train_accu": tr_accuracy,
                "train_loss": tr_loss,
                "test_accu": ts_accuracy,
                "test_loss": ts_loss,
                "attacking": False,
                "client_type": self.client_type,
            },
        )

    def perform_evaluations(self, model, device, trainloader=None, testloader=None):
        
        # Check if data loaders need to be created
        if trainloader is None:
            trainloader = torch.utils.data.DataLoader(self._trainset, batch_size=1024, shuffle=False)

        if testloader is None:
            testloader = torch.utils.data.DataLoader(self._testset, batch_size=1024, shuffle=False)

        # model = model.to(device)

        # Perform necessary evaluations
        tr_loss, tr_accuracy, _ = modules.evaluate(model, trainloader, device=device)
        ts_loss, ts_accuracy, _ = modules.evaluate(model, testloader, device=device)

        # Performing cleanups
        del trainloader, testloader
        
        # Return evaluation stats
        return tr_loss, tr_accuracy, ts_loss, ts_accuracy

    def evaluate(self, model, device, ins: EvaluateIns) -> EvaluateRes:
        config = ins.config

        # Get training config
        server_round = int(ins.config["server_round"])
        total_rounds = int(ins.config["total_rounds"])
        batch_size = int(config["batch_size"])
        criterion_str = config["criterion"]

        # Use provided weights to update the local model
        model.set_weights(ins.parameters, clone=(not self._process))
        model.to(device)

        # Evaluate the updated model on the local dataset
        testloader = torch.utils.data.DataLoader(
            self._testset, batch_size=batch_size, shuffle=False
        )
        criterion = modules.get_criterion(
            criterion_str=criterion_str
        )

        # Collect return results
        loss, accuracy, num_examples = modules.evaluate(model, testloader, device=device, criterion=criterion)

        # Performing cleanups
        del testloader
 
        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=num_examples,
            metrics={
                "accuracy": float(accuracy),
                "loss": float(loss),
            },
        )
