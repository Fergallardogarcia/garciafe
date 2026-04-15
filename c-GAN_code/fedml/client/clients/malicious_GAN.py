"""Implementation of Honest Client using FedML Framework"""

import copy
import json
import timeit
from xml.parsers.expat import model
from fedml import modules
from fedml.common.typing import Code
from fedml.defenses.filters.gan_filter import generate_dataset
from fedml.modules import evaluate_gan



import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader

import numpy as np

import torch
from torch.utils.data import Dataset

from logging import DEBUG, INFO
from typing import Optional, Dict, Tuple
from fedml.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Status,
    log,
)

from .honest_client import HonestClient
from models.model import BaseModel
from models.model_loader import load_model 

class GanMaliciousClient(HonestClient):
    """A malicious client submitting updates with flipped gradient signs.
    
    """
    def __init__(
            self, 
            client_id: int,
            trainset: Dataset,
            testset: Dataset,
            process: bool = True,
            attack_config: Optional[Dict] = None,
            gen_model: Optional[BaseModel] = None,
            initial_parameters: Optional[torch.Tensor] = None,
        ) -> None:
        """Initializes a new client."""
        super().__init__(
            client_id=client_id,
            trainset=trainset,
            testset=testset,
            process=process,
            initial_parameters=initial_parameters,
        )
        #--------------------------------------
        self.attack_config = copy.deepcopy(attack_config)
        filter_param = self.attack_config["GAN_ATTACK_CONFIG"]["HYPER_PARAM"]
        self.synth_strength_ratio = self.attack_config["GAN_ATTACK_CONFIG"]["HYPER_PARAM"]["SYNTHETIC_STRENGTH_RATIO"]
        self.num_synth_samples = self.attack_config["GAN_ATTACK_CONFIG"]["HYPER_PARAM"]["SYNTHETIC_DATA_SIZE"]

        if self.num_synth_samples == "trainset_size":
            self.num_synth_samples = len(trainset)
        
        
        if gen_model is None:
            self.gen_model = load_model(attack_config["GAN_ATTACK_CONFIG"]["HYPER_PARAM"]["GEN_ARGS"])
        else:
            self.gen_model = gen_model
        

    @property
    def client_type(self):
        """Returns current client's type."""
        return "GAN_ATTACK"



    
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

        gen_configs= self.attack_config["GAN_ATTACK_CONFIG"]["HYPER_PARAM"]["GEN_ARGS"]
        num_classes= gen_configs["NUM_CLASSES"]
        latent_size = gen_configs["LATENT_SIZE"]


        attack = np.random.random() < self.attack_config["ATTACK_RATIO"]

        if server_round < self.attack_config["ATTACK_ROUND"] or not attack:
            # return super().fit(model=model, device=device, ins=ins)
            #For returning just the global model parameters without any local training: 
            return super().fit(model=model, device=device, ins=ins)

        # Train an honest reference update from the same starting point.
        honest_fit_res = super().fit(model=model, device=device, ins=ins)
        honest_parameters = honest_fit_res.parameters.detach().clone()
        initial_parameters = ins.parameters.detach().clone()

        # Set model parameters
        model.set_weights(ins.parameters, clone=(not self._process))
        model.to(device)
        # Set Gen model to device for client:
        self.gen_model.to(device)


        # #switch datasets for the training. From real to mixed (real + synthetic)
        # original_trainset = self._trainset
        # self._trainset = mixed_trainset



        # Stage dataset to GPU
        original_device = self._trainset.data.device
        self._trainset.to_device(device=device)

        # Train model
        trainloader = torch.utils.data.DataLoader(
            self._trainset, batch_size=batch_size, shuffle=True, drop_last=False
        )


        # # Train model
        # trainloader = torch.utils.data.DataLoader(
        #     mixed_trainset, batch_size=batch_size, shuffle=True, drop_last=False
        # )

        criterion = modules.get_criterion(
            criterion_str=criterion_str 
        )
        #, reduction="none"

        training_mode = str(
            self.attack_config["GAN_ATTACK_CONFIG"]["HYPER_PARAM"].get("DISCRIMINATOR_TRAINING_MODE", "sequential")
        ).lower()
        if training_mode not in {"pcgrad", "sequential"}:
            log(
                INFO,
                f"Unknown DISCRIMINATOR_TRAINING_MODE={training_mode}, falling back to sequential",
            )
            training_mode = "sequential"
        global_model = model if training_mode == "sequential" else None
        if training_mode == "sequential":
            from fedml.models.resnet_custom import ResNet18
            model = ResNet18(num_classes=num_classes).to(device)
        
        
        optimizer = modules.get_optimizer(
            optimizer_str=optimizer_str,            
            local_model=model,
            learning_rate=learning_rate,
            **optim_kwargs,
        )

        

        # train_malgan_fn = (
        #     train_malGAN_discriminator_sequential
        #     if training_mode == "sequential"
        #     else train_malGAN_discriminator
        # )
        train_malgan_fn= train_alternate

        
        

        num_examples, avg_grad_cosine_similarity = train_malgan_fn(
            model=model,
            gen_model= self.gen_model,
            global_model=global_model,
            trainloader=trainloader, 
            epochs=local_epochs, 
            learning_rate=learning_rate,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_classes=num_classes,
            latent_size=latent_size,
            synth_strength_ratio=self.synth_strength_ratio,
            priority_loss=gen_configs.get("PCGRAD_PRIORITY_LOSS", "fake"),
            grad_other_scale=float(
                gen_configs.get(
                    "PCGRAD_GRAD_OTHER_SCALE",self.synth_strength_ratio,
                )
            ),
            kd_alpha=float(gen_configs.get("KD_ALPHA", 1.0)),
            kd_temperature=float(gen_configs.get("KD_TEMPERATURE", 5.0)),
        )

        # num_examples = train_mixed_data(
        #     model=model, 
        #     trainloader=trainloader, 
        #     epochs=local_epochs, 
        #     learning_rate=learning_rate,
        #     criterion=criterion,
        #     optimizer=optimizer,
        #     device=device,
        #     coeff_real=coeff_real,
        #     coeff_synth=coeff_synth,
        #     synth_strength_ratio=self.synth_strength_ratio,
        # )

        # Get weights from the model and stage back to CPU if running as process
        parameters_updated = model.get_weights()
        if self._process: parameters_updated = parameters_updated.cpu()

        malicious_update = parameters_updated - initial_parameters
        honest_update = honest_parameters - initial_parameters

        honest_norm_sq = torch.dot(honest_update, honest_update)
        if honest_norm_sq.item() > 1e-12:
            proj_coeff = torch.dot(malicious_update, honest_update) / honest_norm_sq
            projection_on_honest = proj_coeff * honest_update
        else:
            projection_on_honest = torch.zeros_like(malicious_update)
        perpendicular_to_honest = malicious_update - projection_on_honest

        honest_update_norm = torch.linalg.vector_norm(honest_update).item()
        malicious_update_norm = torch.linalg.vector_norm(malicious_update).item()

        cosine_honest_malicious= torch.dot(honest_update, malicious_update) / (honest_update_norm * malicious_update_norm)
        projection_on_honest_norm = torch.linalg.vector_norm(projection_on_honest).item()
        perpendicular_to_honest_norm = torch.linalg.vector_norm(perpendicular_to_honest).item()

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


        #Stage client gen model back to original device----------
        self.gen_model.to(original_device)

        # # Restore original dataset
        # self._trainset = original_trainset

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
                "attacking": True,
                "client_type": self.client_type,
                "avg_grad_cosine_similarity": float(avg_grad_cosine_similarity),
                "honest_update_norm": float(honest_update_norm),
                "malicious_update_norm": float(malicious_update_norm),
                "cosine_honest_malicious_update": float(cosine_honest_malicious),
                "malicious_update_proj_honest_norm": float(projection_on_honest_norm),
                "malicious_update_perp_honest_norm": float(perpendicular_to_honest_norm),
            },
        )

def train_malGAN_discriminator(
        model: nn.Module,
        gen_model: nn.Module,
        global_model: Optional[nn.Module],
        trainloader: torch.utils.data.DataLoader,
        epochs: int,
        device: str,  # pylint: disable=no-member
        learning_rate: float,
        criterion,
        optimizer,
        num_classes: int,
        latent_size: int,
        synth_strength_ratio: float,
        priority_loss: str = "fake",
        grad_other_scale: float = 1.0,
        kd_alpha: float = 1.0,
        kd_temperature: float = 5.0,
    ) -> Tuple[int, float]:

    def _flatten_grads(grads):
        return torch.cat([grad.reshape(-1) for grad in grads])

    def _cosine_similarity_between_grads(grads_a, grads_b, eps: float = 1e-12) -> float:
        flat_a = _flatten_grads(grads_a)
        flat_b = _flatten_grads(grads_b)
        norm_a = torch.linalg.vector_norm(flat_a)
        norm_b = torch.linalg.vector_norm(flat_b)
        if norm_a.item() <= 0 or norm_b.item() <= 0:
            return 0.0
        cosine_similarity = torch.dot(flat_a, flat_b) / (norm_a * norm_b).clamp_min(eps)
        return float(cosine_similarity.item())
    
    def _merge_two_loss_grads_with_pcgrad(
        model: nn.Module,
        loss_real: torch.Tensor,
        loss_fake: torch.Tensor,
        priority_loss: str,
        grad_other_scale: float,
        eps: float = 1e-12,
    ):
        """Project conflicting gradients while prioritizing one objective."""
        params = [param for param in model.parameters() if param.requires_grad]
        if not params:
            return [], [], 0.0

        grads_real = torch.autograd.grad(
            loss_real,
            params,
            retain_graph=True,
            allow_unused=True,
        )
        grads_fake = torch.autograd.grad(
            loss_fake,
            params,
            retain_graph=False,
            allow_unused=True,
        )

        grads_real = [
            grad.detach().clone() if grad is not None else torch.zeros_like(param)
            for grad, param in zip(grads_real, params)
        ]
        grads_fake = [
            grad.detach().clone() if grad is not None else torch.zeros_like(param)
            for grad, param in zip(grads_fake, params)
        ]

        grad_cosine_similarity = _cosine_similarity_between_grads(
            grads_a=grads_real,
            grads_b=grads_fake,
            eps=eps,
        )

        flat_real = torch.cat([grad.reshape(-1) for grad in grads_real])
        flat_fake = torch.cat([grad.reshape(-1) for grad in grads_fake])

        if priority_loss not in {"real", "fake"}:
            raise ValueError("priority_loss must be either 'real' or 'fake'")
        grad_other_scale = max(float(grad_other_scale), 0.0)

        if priority_loss == "real":
            flat_priority, flat_other = flat_real, flat_fake
            grads_priority, grads_other = grads_real, grads_fake
        else:
            flat_priority, flat_other = flat_fake, flat_real
            grads_priority, grads_other = grads_fake, grads_real

        dot_product = torch.dot(flat_priority, flat_other)
        if dot_product.item() < 0:
            priority_norm_sq = torch.dot(flat_priority, flat_priority).clamp_min(eps)
            proj_grads_other = [
                grad_other - (dot_product / priority_norm_sq) * grad_priority 
                for grad_priority, grad_other in zip(grads_priority, grads_other)
            ]
        else:
            proj_grads_other = grads_other

        merged_grads = [
            grad_priority + grad_other_scale * grad_other
            for grad_priority, grad_other in zip(grads_priority, proj_grads_other)
        ]
        return params, merged_grads, grad_cosine_similarity

    num_examples = 0
    cosine_similarity_sum = 0.0
    cosine_similarity_count = 0
    model.train()
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_real_loss = 0.0
        running_fake_loss = 0.0
        for i, data in enumerate(trainloader):
            images, labels = data[0].to(device), data[1].to(device)

            # Labels
            real_labels = labels
            fake_labels = labels

            optimizer.zero_grad()

            # --- Build both objectives first, then apply PCGrad ---
            real_outputs = model(images)
            real_loss = - criterion(real_outputs, real_labels) # (1 - synth_strength_ratio) 

            # gen_model(current_z, current_l)
            input_znoises = torch.randn(labels.size(0), latent_size).to(device)
            
            fake_data = gen_model(input_znoises, fake_labels).detach()  # detach to avoid training generator here
            fake_outputs = model(fake_data)
            fake_loss =  criterion(fake_outputs, fake_labels) # synth_strength_ratio 

            params, merged_grads, grad_cosine_similarity = _merge_two_loss_grads_with_pcgrad(
                model=model,
                loss_real=real_loss,
                loss_fake=fake_loss,
                priority_loss=priority_loss,
                grad_other_scale=grad_other_scale,
            )

            for param, grad in zip(params, merged_grads):
                param.grad = grad

            # Update discriminator
            optimizer.step()

            # print statistics
            running_real_loss += real_loss.item()
            running_fake_loss += fake_loss.item()
            num_examples += labels.size(0)
            cosine_similarity_sum += grad_cosine_similarity
            cosine_similarity_count += 1

    avg_grad_cosine_similarity = cosine_similarity_sum / max(cosine_similarity_count, 1)
    return num_examples, avg_grad_cosine_similarity


def train_malGAN_discriminator_sequential(
        model: nn.Module,
        gen_model: nn.Module,
        global_model: Optional[nn.Module],
        trainloader: torch.utils.data.DataLoader,
        epochs: int,
        device: str,  # pylint: disable=no-member
        learning_rate: float,
        criterion,
        optimizer,
        num_classes: int,
        latent_size: int,
        synth_strength_ratio: float,
        priority_loss: str = "fake",
        grad_other_scale: float = 1.0,
        kd_alpha: float = 1.0,
        kd_temperature: float = 5.0,
    ) -> Tuple[int, float]:
    """Alternative discriminator training with two sequential updates.

    For each batch: first optimize using real_loss, then optimize using fake_loss.
    The return type matches train_malGAN_discriminator for easy swapping.
    """


    if global_model is not None:
        global_model.eval()

    def _flatten_grads(grads):
        return torch.cat([grad.reshape(-1) for grad in grads])

    def _cosine_similarity_between_grads(grads_a, grads_b, eps: float = 1e-12) -> float:
        flat_a = _flatten_grads(grads_a)
        flat_b = _flatten_grads(grads_b)
        norm_a = torch.linalg.vector_norm(flat_a)
        norm_b = torch.linalg.vector_norm(flat_b)
        if norm_a.item() <= 0 or norm_b.item() <= 0:
            return 0.0
        cosine_similarity = torch.dot(flat_a, flat_b) / (norm_a * norm_b).clamp_min(eps)
        return float(cosine_similarity.item())

    num_examples = 0
    cosine_similarity_sum = 0.0
    cosine_similarity_count = 0
    params = [param for param in model.parameters() if param.requires_grad]

    model.train()
    for epoch in range(5*epochs):
        running_real_loss = 0.0
        running_fake_loss = 0.0
        for i, data in enumerate(trainloader):
            images, labels = data[0].to(device), data[1].to(device)

            # # Step 1: optimize with real loss.
            # optimizer.zero_grad()
            # real_outputs = model(images)
            # real_loss = - criterion(real_outputs, labels)
            # real_loss.backward()
            # real_grads = [
            #     param.grad.detach().clone() if param.grad is not None else torch.zeros_like(param)
            #     for param in params
            # ]
            # optimizer.step()

            # Step 2: optimize with fake loss.
           
            optimizer.zero_grad()
            input_znoises = torch.randn(labels.size(0), latent_size).to(device)
            fake_data = gen_model(input_znoises, labels).detach()
            fake_outputs = model(fake_data)
            fake_loss = criterion(fake_outputs, labels)

            if global_model is not None:
                with torch.no_grad():
                    teacher_outputs = global_model(fake_data)
                kd_loss = F.kl_div(
                    F.log_softmax(fake_outputs / kd_temperature, dim=1),
                    F.softmax(teacher_outputs / kd_temperature, dim=1),
                    reduction="batchmean",
                ) * (kd_temperature ** 2)
                total_loss = (1.0 - kd_alpha) * fake_loss + kd_alpha * kd_loss
            else:
                total_loss = fake_loss

            total_loss.backward()
            fake_grads = [
                param.grad.detach().clone() if param.grad is not None else torch.zeros_like(param)
                for param in params
            ]
            optimizer.step()

            # grad_cosine_similarity = _cosine_similarity_between_grads(
            #     grads_a=real_grads,
            #     grads_b=fake_grads,
            # )

            # running_real_loss += real_loss.item()
            running_fake_loss += fake_loss.item()
            num_examples += labels.size(0)
            # cosine_similarity_sum += grad_cosine_similarity
            cosine_similarity_count += 1

    avg_grad_cosine_similarity = cosine_similarity_sum / max(cosine_similarity_count, 1)
    return num_examples, avg_grad_cosine_similarity



def train_alternate(
        model: nn.Module,
        gen_model: nn.Module,
        global_model: Optional[nn.Module],
        trainloader: torch.utils.data.DataLoader,
        epochs: int,
        device: str,  # pylint: disable=no-member
        learning_rate: float,
        criterion,
        optimizer,
        num_classes: int,
        latent_size: int,
        synth_strength_ratio: float,
        priority_loss: str = "fake",
        grad_other_scale: float = 1.0,
        kd_alpha: float = 1.0,
        kd_temperature: float = 5.0,
    ) -> Tuple[int, float]:
    """Helper function to train the model.

    :param model: The local model that needs to be trained.
    :param trainloader: The dataloader of the dataset to use for training.
    :param epochs: Number of training rounds / epochs
    :param device: The device to train the model on i.e. cpu or cuda. 
    :param learning_rate: The initial learning rate the optimizer is using.
    :param criterion: The loss function to use for model training.
    :param optimizer: The optimizer to use for model training.
    :param coeff_real: The coefficient for real data loss.
    :param coeff_synth: The coefficient for synthetic data loss.
    :returns: A tuple containing the number of examples and the average cosine similarity.
    """
    if global_model is not None:
        global_model.eval()

    num_examples = 0
    model.train()

    # Phase 1: full training on real samples.
    for epoch in range(epochs):
        running_real_loss = 0.0
        for i, data in enumerate(trainloader):
            images, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            real_loss = - criterion(model(images), labels)
            real_loss.backward()
            optimizer.step()

            running_real_loss += real_loss.item()
            num_examples += labels.size(0)

    # Restart optimizer state before fake/KD phase.
    optimizer.state.clear()

    # Phase 2: full training on fake samples with KD.
    for epoch in range(epochs):
        running_fake_loss = 0.0
        for i, data in enumerate(trainloader):
            _, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            input_znoises = torch.randn(labels.size(0), latent_size).to(device)
            fake_data = gen_model(input_znoises, labels).detach()
            fake_outputs = model(fake_data)
            fake_loss = criterion(fake_outputs, labels)

            if global_model is not None:
                with torch.no_grad():
                    teacher_outputs = global_model(fake_data)
                kd_loss = F.kl_div(
                    F.log_softmax(fake_outputs / kd_temperature, dim=1),
                    F.softmax(teacher_outputs / kd_temperature, dim=1),
                    reduction="batchmean",
                ) * (kd_temperature ** 2)
                total_loss = (1.0 - kd_alpha) * fake_loss + kd_alpha * kd_loss
            else:
                total_loss = fake_loss

            total_loss.backward()
            optimizer.step()

            running_fake_loss += fake_loss.item()
            num_examples += labels.size(0)

    return num_examples, 0.0



