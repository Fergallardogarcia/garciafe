"""Implementation of Honest Client using FedML Framework"""

import copy
import json
import math
import timeit
from xml.parsers.expat import model
from fedml import modules
from fedml.common.typing import Code
from fedml.defenses.filters.gan_filter import generate_dataset
from fedml.modules import evaluate, evaluate_gan
from fedml.modules import get_optimizer


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import TensorDataset, DataLoader
from fedml.models.resnet_custom import ResNet18

import numpy as np

import torch
from torch.utils.data import Dataset

from logging import DEBUG, INFO
from typing import Optional, Dict, Tuple, cast
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
from .correction_diagnostics import CorrectionDiagnostics
from models.model import BaseModel
from models.model_loader import load_model
from . import malicious_random as malicious_random_attack

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
        self.gan_config = self.attack_config["GAN_ATTACK_CONFIG"]["HYPER_PARAM"]
        self.synth_strength_ratio = self.attack_config["GAN_ATTACK_CONFIG"]["HYPER_PARAM"]["SYNTHETIC_STRENGTH_RATIO"]
        self.num_synth_samples = self.attack_config["GAN_ATTACK_CONFIG"]["HYPER_PARAM"]["SYNTHETIC_DATA_SIZE"]

        if self.num_synth_samples == "trainset_size":
            self.num_synth_samples = len(trainset)
        
        
        if gen_model is None:
            self.gen_model = load_model(attack_config["GAN_ATTACK_CONFIG"]["HYPER_PARAM"]["GEN_ARGS"])
        else:
            self.gen_model = gen_model
        self.gen_def_dataset=None
        self.gen_dataset = None
        

    @property
    def client_type(self):
        """Returns current client's type."""
        return "GAN_ATTACK"



    
    def fit(self, model, device, ins: FitIns) -> FitRes:
        config = ins.config
        fit_begin = timeit.default_timer()
        status = Status(code=Code.OK, message="Success")
        fitres = FitRes(
            status=status,
            parameters=ins.parameters,
            num_examples=0,
            metrics={
                "client_id": int(self.client_id),
                "attacking": True,
                "client_type": self.client_type,
            },
        )

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


        # Extract KD config (allow either direct keys or nested KD_CONFIG)
        kd_cfg = self.gan_config.get("KD_CONFIG", {}) if isinstance(self.gan_config, dict) else {}
        kd_alpha_val = float(kd_cfg.get("ALPHA", self.gan_config.get("KD_ALPHA", 1.0)))
        kd_temperature_val = float(kd_cfg.get("TEMPERATURE", self.gan_config.get("KD_TEMPERATURE", 7.0)))

        #----------------------------------------
        

        attack = np.random.random() < self.attack_config["ATTACK_RATIO"]

        if server_round < self.attack_config["ATTACK_ROUND"] or not attack:
            # return super().fit(model=model, device=device, ins=ins)
            #For returning just the global model parameters without any local training: 
            return super().fit(model=model, device=device, ins=ins)

        # Train an honest reference update from the same starting point.
        honest_fit_res = super().fit(model=model, device=device, ins=ins)
        honest_parameters = honest_fit_res.parameters.detach().clone()
        initial_parameters = ins.parameters.detach().clone()

        honest_update = honest_parameters - initial_parameters
        honest_update_norm = torch.linalg.vector_norm(honest_update).item()
        if ins.gan_attack_payload is not None:
            ins.gan_attack_payload.perturbation_magnitude = honest_update_norm

        # Set model parameters (after honest_fit, that modified it)
        model.set_weights(ins.parameters, clone=(not self._process))
        model.to(device)
        # Set Gen model to device for client:
        self.gen_model.to(device)

        #Training type
        training_mode = str(
            self.gan_config.get("DISCRIMINATOR_TRAINING_MODE", "sequential")
        ).lower()
        if training_mode not in {"pcgrad", "sequential"}:
            training_mode = "other"

        

        # Keep a global reference model for KD and loss-ratio logging.
        global_model = copy.deepcopy(model)
        global_model.set_weights(ins.parameters, clone=(not self._process))
        global_model.to(device)

        # Displace parameters before synthetic training.
        if training_mode == "sequential":

            
            # model = ResNet18(num_classes=num_classes).to(device)
            
            from .malicious_signflip import SignFlipClient
            Sf_cl= SignFlipClient(
                client_id=self.client_id,
                trainset=self._trainset,
                testset=self._testset,
                process=self._process,
                attack_config=self.attack_config,
            )
            Sf_fitres = Sf_cl.fit(model=model, device=device, ins=ins)  
            model.set_weights(Sf_fitres.parameters, clone=(not self._process))

        

        # Stage dataset to GPU
        original_device = self._trainset.data.device
        self._trainset.to_device(device=device)

        # Train model
        trainloader = torch.utils.data.DataLoader(
            self._trainset, batch_size=batch_size, shuffle=True, drop_last=False
        )


        # # Train model

        criterion = modules.get_criterion(
            criterion_str=criterion_str 
        )
        #, reduction="none"

        

        
        optimizer = modules.get_optimizer(
            optimizer_str=optimizer_str,
            local_model=model,
            learning_rate=learning_rate,
            **optim_kwargs,
        )

        # Reference loss for perturbation tuning: the attacker-generator KD loss of the
        # honest-trained model (teacher = global model). Stored on the payload so the
        # correction step uses the honest update -- not the global model -- as the baseline.
        if ins.gan_attack_payload is not None:
            self.gen_model.eval()
            with torch.no_grad():
                honest_eval_labels = honest_eval_noise = None
                for _hdata in trainloader:
                    honest_eval_labels = _hdata[1].to(device)
                    honest_eval_noise = torch.randn(honest_eval_labels.size(0), latent_size).to(device)
                    break
                if honest_eval_labels is not None:
                    saved_weights = model.get_weights()
                    model.set_weights(honest_parameters)
                    model.eval()
                    honest_images = self.gen_model(honest_eval_noise, honest_eval_labels)
                    honest_teacher = global_model(honest_images)
                    honest_loss_value = compute_kd_loss(
                        student_outputs=model(honest_images),
                        labels=honest_eval_labels,
                        criterion=criterion,
                        teacher_outputs=honest_teacher,
                        kd_alpha=0.0,
                        kd_temperature=kd_temperature_val,
                    )
                    model.set_weights(saved_weights)
                    ins.gan_attack_payload.honest_loss = float(honest_loss_value.item())

        # train_malgan_fn = (
        #     train_malGAN_discriminator_sequential
        #     if training_mode == "sequential"
        #     else train_malGAN_discriminator
        # )
        train_malgan_fn = perturbation_plus_correction

        

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
            priority_loss=self.gan_config.get("PCGRAD_PRIORITY_LOSS", "fake"),
            grad_other_scale=float(
                self.gan_config.get(
                    "PCGRAD_GRAD_OTHER_SCALE",self.synth_strength_ratio,
                )
            ),
            kd_alpha=kd_alpha_val,
            kd_temperature=kd_temperature_val,
            fitres=fitres,
            fitins=ins,
            attack_config=self.attack_config,
        )

        # Get weights from the model and stage back to CPU if running as process
        parameters_updated = model.get_weights()
        if self._process: parameters_updated = parameters_updated.cpu()

        malicious_update = parameters_updated - initial_parameters

        honest_norm_sq = torch.dot(honest_update, honest_update)
        if honest_norm_sq.item() > 1e-12:
            proj_coeff = torch.dot(malicious_update, honest_update) / honest_norm_sq
            projection_on_honest = proj_coeff * honest_update
        else:
            projection_on_honest = torch.zeros_like(malicious_update)
        perpendicular_to_honest = malicious_update - projection_on_honest

        malicious_update_norm = torch.linalg.vector_norm(malicious_update).item()

        cosine_honest_malicious= torch.dot(honest_update, malicious_update) / (honest_update_norm * malicious_update_norm)
        projection_on_honest_norm = torch.linalg.vector_norm(projection_on_honest).item()
        perpendicular_to_honest_norm = torch.linalg.vector_norm(perpendicular_to_honest).item()

        fit_duration = timeit.default_timer() - fit_begin

        # Perform necessary evaluations
        ts_loss, ts_accuracy, tr_loss, tr_accuracy = (0.0, 0.0, 0.0, 0.0)
        if perform_evals:
            tr_loss, tr_accuracy, ts_loss, ts_accuracy = self.perform_evaluations(
                model,
                device,
                trainloader=None,
                testloader=None,
            )
            _, _, gl_ts_loss, _ = self.perform_evaluations(
                global_model,
                device,
                trainloader=None,
                testloader=None,
            )

            if abs(gl_ts_loss) > 1e-12:
                test_loss_ratio = ts_loss / gl_ts_loss
            elif abs(ts_loss) <= 1e-12:
                test_loss_ratio = 1.0
            else:
                test_loss_ratio = float("inf")



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
        fitres.status = status
        fitres.parameters = parameters_updated
        fitres.num_examples = num_examples
        fitres.metrics["client_id"] = int(self.client_id)
        fitres.metrics["fit_duration"] = fit_duration
        fitres.metrics["train_accu"] = tr_accuracy
        fitres.metrics["train_loss"] = tr_loss
        fitres.metrics["test_accu"] = ts_accuracy
        fitres.metrics["test_loss"] = ts_loss
        fitres.metrics["attacking"] = True
        fitres.metrics["client_type"] = self.client_type


        if bool(self.gan_config.get("TEST", False)):
            fitres.parameters = honest_parameters

        return fitres




def optimize_norm_scale(
        model: nn.Module,
        gen_model: nn.Module,
        global_model: Optional[nn.Module],
        trainloader: torch.utils.data.DataLoader,
        model_weights,
        direction_tensor,
        base_magnitude: float,
        norm_scale: float,
        device: str,
        latent_size: int,
        criterion,
        kd_alpha: float,
        kd_temperature: float,
        target_ratio: float = 2.0,
        tolerance: float = 0.5,
        min_norm_scale: float = 1.0,
        max_norm_scale: float = 10.0,
        max_evaluations: int = 10,
        max_perturbed_loss: float = 0.5,
        honest_loss: Optional[float] = None,
    ) -> Tuple[float, float, float, float, int]:
    """Tune the perturbation ``norm_scale`` so the attacker-generator loss ratio
    ``loss_perturbed / loss_honest`` is as close to ``target_ratio`` as possible, subject to a
    hard cap ``loss_perturbed <= max_perturbed_loss`` and ``norm_scale in [min, max]``.

    Both losses are the classification loss (``criterion``) of the model on the attacker's
    generator samples. The applied perturbation is ``base_magnitude * norm_scale *
    direction_tensor`` -- only ``norm_scale`` is optimized. The objective reduces to a 1-D
    monotone root-find: pick the target loss ``goal = min(target_ratio * loss_honest,
    max_perturbed_loss)`` (so the cap always wins) and find ``norm_scale`` with
    ``loss_perturbed(norm_scale) = goal``. We bracket with the two box endpoints and run an
    Illinois (modified false-position) search in log-log space -- derivative-free,
    bracket-preserving (no overshoot), superlinear -- within ``max_evaluations`` loss
    evaluations. The returned point is the evaluated ``norm_scale`` whose loss is closest to
    ``goal`` while staying ``<= max_perturbed_loss`` (if no evaluated point satisfies the cap,
    the lowest-loss one is returned as best effort). ``norm_scale`` is always within
    ``[min_norm_scale, max_norm_scale]``; ``model`` is left at the corresponding perturbed
    weights. The honest reference loss is taken from ``honest_loss`` when provided (the loss of
    the honest-trained model, attached to the attack payload); otherwise it is measured on the
    pre-perturbation model here. Returns
    ``(norm_scale, perturbation_magnitude, loss_honest, loss_perturbed, evaluations)`` where
    ``perturbation_magnitude = base_magnitude * norm_scale``.
    """
    gen_model.eval()
    # Synthetic batch from the attacker generator for the perturbed-loss evaluations.
    eval_noise = None
    eval_labels = None
    for _data in trainloader:
        eval_labels = _data[1].to(device)
        eval_noise = torch.randn(eval_labels.size(0), latent_size).to(device)
        break

    def _attacker_gen_loss() -> float:
        if eval_noise is None:
            return 0.0
        model.eval()
        with torch.no_grad():
            images = gen_model(eval_noise, eval_labels)
            teacher_outputs = global_model(images) if global_model is not None else None
            loss = compute_kd_loss(
                student_outputs=model(images),
                labels=eval_labels,
                criterion=criterion,
                teacher_outputs=teacher_outputs,
                kd_alpha=0.0,
                kd_temperature=kd_temperature,
            )
        return float(loss.item())

    eps = 1e-12  # loss floor so the log-space search and the ratio stay finite.

    with torch.no_grad():
        # Honest reference loss: prefer the precomputed payload value (loss of the honest-trained
        # model); only fall back to measuring the pre-perturbation model here.
        if honest_loss is None:
            model.set_weights(model_weights)
            loss_honest = max(_attacker_gen_loss(), eps)
            evaluations = 1
        else:
            loss_honest = max(honest_loss, eps)
            evaluations = 0

        def loss_at(scale: float) -> float:
            nonlocal evaluations
            model.set_weights(model_weights + (base_magnitude * scale) * direction_tensor)
            evaluations += 1
            return max(_attacker_gen_loss(), eps)

        # Target perturbed loss: aim for the ratio target, but never above the hard cap.
        ratio_target_loss = target_ratio * loss_honest
        goal = min(ratio_target_loss, max_perturbed_loss)
        # Convergence band: the ratio tolerance when the ratio target is achievable; otherwise a
        # tight band just under the cap (we want loss as close to the cap as allowed).
        if ratio_target_loss <= max_perturbed_loss:
            loss_tol = tolerance * loss_honest
        else:
            loss_tol = 0.05 * max_perturbed_loss

        # Track the best feasible point (loss <= cap, closest to goal) and an overall
        # lowest-loss fallback for the case where no evaluated point satisfies the cap.
        best_s = best_loss = None
        low_s = low_loss = None

        def record(scale: float, loss: float) -> None:
            nonlocal best_s, best_loss, low_s, low_loss
            if low_s is None or loss < low_loss:
                low_s, low_loss = scale, loss
            if loss <= max_perturbed_loss + 1e-9:
                if best_s is None or abs(loss - goal) < abs(best_loss - goal):
                    best_s, best_loss = scale, loss

        # Bracket with the box endpoints (loss is monotone in scale, either direction).
        a, b = min_norm_scale, max_norm_scale
        la = loss_at(a); record(a, la)
        lb = loss_at(b); record(b, lb)

        # Illinois false-position in log-log space, but only if the goal is actually bracketed.
        xa, xb = math.log(a), math.log(b)
        fa = math.log(la) - math.log(goal)
        fb = math.log(lb) - math.log(goal)
        if (fa < 0.0) != (fb < 0.0):
            while evaluations < max_evaluations:
                if best_loss is not None and abs(best_loss - goal) <= loss_tol:
                    break
                xc = xb - fb * (xb - xa) / (fb - fa)
                sc = math.exp(xc)
                lc = loss_at(sc); record(sc, lc)
                fc = math.log(lc) - math.log(goal)
                if (fc < 0.0) == (fa < 0.0):
                    xa, fa = xc, fc
                    fb *= 0.5  # Illinois: dampen the stale endpoint to avoid stalling.
                else:
                    xb, fb = xc, fc
                    fa *= 0.5

        # Prefer the best feasible point; otherwise the lowest-loss one (closest to the cap).
        if best_s is not None:
            norm_scale, loss_perturbed = best_s, best_loss
        else:
            norm_scale, loss_perturbed = low_s, low_loss

        norm_scale = min(max(norm_scale, min_norm_scale), max_norm_scale)
        model.set_weights(model_weights + (base_magnitude * norm_scale) * direction_tensor)

    return norm_scale, base_magnitude * norm_scale, loss_honest, loss_perturbed, evaluations


def perturbation_plus_correction(
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
        fitres: FitRes,
        fitins: FitIns,
        priority_loss: str = "fake",
        grad_other_scale: float = 1.0,
        kd_alpha: float = 0.0,
        kd_temperature: float = 8.0,
        attack_config: Optional[Dict] = None,
    ) -> Tuple[int, float]:
    """Helper function to train the model on synthetic data only."""
    if global_model is not None:
        global_model.eval()
    if fitins is None:
        raise ValueError("perturbation_plus_correction requires FitIns")

    attack_payload = fitins.gan_attack_payload
    perturbation_direction = attack_payload.perturbation_direction if attack_payload is not None else None
    if perturbation_direction is None:
        raise ValueError("perturbation_plus_correction requires gan_attack_payload.perturbation_direction")

    num_examples = 0
    model.to(device)
    model.eval()

    fitres_metrics = None
    if fitres is not None:
        fitres_metrics = cast(Dict[str, object], fitres.metrics)
    client_id_for_log = "-"
    if fitres_metrics is not None and "client_id" in fitres_metrics:
        client_id_for_log = str(fitres_metrics["client_id"])

    fitres.param_array["pre"] = model.get_weights()
    
    # Perturbation = base_magnitude (honest update norm) * norm_scale, along direction_tensor.
    # Only norm_scale is optimized during correction tuning.
    base_magnitude = None
    norm_scale = float(attack_config["RANDOM_CONFIG"]["NORM_SCALE"])
    perturbation_magnitude = None
    if attack_payload is not None and attack_payload.perturbation_magnitude is not None:
        base_magnitude = attack_payload.perturbation_magnitude
        perturbation_magnitude = base_magnitude * norm_scale

    with torch.no_grad():
        model_weights = model.get_weights()
        direction_tensor = perturbation_direction.to(
            model_weights.device,
            dtype=model_weights.dtype,
        )
        if direction_tensor.shape != model_weights.shape:
            raise ValueError("perturbation_direction shape does not match model.get_weights()")
        model.set_weights(model_weights + perturbation_magnitude * direction_tensor)

    correction_direction = direction_tensor.detach()
    
    honest_update_direction = None
    honest_update_sign = None
    if attack_payload is not None:
        honest_update_direction = getattr(attack_payload, "honest_update_direction", None)
        honest_update_sign = getattr(attack_payload, "honest_update_sign", None)
    
    # Random init-----------------------------------------------
    # reset_weights = ResNet18(num_classes=num_classes).get_weights()
    # model.set_weights(reset_weights, clone=(not True))

    #----------------------------------------------------------
    correction_config = attack_config["GAN_ATTACK_CONFIG"]["HYPER_PARAM"]["CORRECTION_CONFIG"]

    # Before correction, tune the perturbation norm_scale so the attacker-generator loss ratio
    # (perturbed / honest) lands within tolerance of the target ratio.
    if correction_config.get("CORRECTION", True) and correction_config.get("TUNE_NORM_SCALE", False):
        assert base_magnitude is not None, "base_magnitude must be set before scaling"
        norm_scale, perturbation_magnitude, loss_honest, loss_perturbed, scale_evals = optimize_norm_scale(
            model=model,
            gen_model=gen_model,
            global_model=global_model,
            trainloader=trainloader,
            model_weights=model_weights,
            direction_tensor=direction_tensor,
            base_magnitude=base_magnitude,
            norm_scale=norm_scale,
            device=device,
            latent_size=latent_size,
            criterion=criterion,
            kd_alpha=0.0,
            kd_temperature=kd_temperature,
            target_ratio=10.0,
            tolerance=2.0,
            min_norm_scale=1.0,
            max_norm_scale=10.0,
            max_evaluations=10,
            max_perturbed_loss=0.5,
            honest_loss=getattr(attack_payload, "honest_loss", None),
        )
        loss_ratio = loss_perturbed / max(loss_honest, 1e-12)
        log(
            INFO,
            "[Client %s] perturbation tuning: loss_honest=%.4f loss_perturbed=%.4f "
            "ratio=%.3f evals=%d norm_scale=%.4f magnitude=%.6f",
            client_id_for_log,
            loss_honest,
            loss_perturbed,
            loss_ratio,
            scale_evals,
            norm_scale,
            perturbation_magnitude,
        )

    fitres.param_array["perturbation"] = model.get_weights()
    if correction_config.get("CORRECTION", True):
        # Step 2: train the perturbed model on synthetic
        model.train()
        gen_model.eval()

        adam_optimizer = get_optimizer( # for constrained opt, used no weight decay. For others unconstrained, using it
            optimizer_str="ADAM",
            local_model=model,
            learning_rate=1e-3,
            # betas=(0.0, 0.9),
        )

        # adam_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)

        # Diagnostics: track whether synthetic-data training is walking the
        # perturbed model back toward the global optimum, and why/why not.
        # diagnostics = None
        # if correction_config.get("DIAGNOSTICS", False):
        #     diagnostics = CorrectionDiagnostics(
        #         model=model,
        #         global_model=global_model,
        #         device=device,
        #         gen_model=gen_model,
        #         criterion=criterion,
        #         loss_fn=compute_kd_loss,
        #         num_classes=num_classes,
        #         latent_size=latent_size,
        #         kd_alpha=kd_alpha,
        #         kd_temperature=kd_temperature,
        #         fresh_size=int(correction_config.get("DIAGNOSTICS_FRESH_SIZE", 256)),
        #         client_id=client_id_for_log,
        #     )

        for epoch in range(epochs):
            for data in trainloader:
                
                pre_weights = model.get_weights()

                labels = data[1].to(device)
                input_znoises = torch.randn(labels.size(0), latent_size).to(device)
                with torch.no_grad():
                    images = gen_model(input_znoises, labels).detach()

                # optimizer.zero_grad()
                adam_optimizer.zero_grad()
                student_outputs = model(images)
                teacher_outputs = None
                if global_model is not None:
                    with torch.no_grad():
                        teacher_outputs = global_model(images)
                total_loss = compute_kd_loss(
                    student_outputs=student_outputs,
                    labels=labels,
                    criterion=criterion,
                    teacher_outputs=teacher_outputs,
                    kd_alpha=kd_alpha,
                    kd_temperature=kd_temperature,
                )

                def _project_gradients_perpendicular_to_direction() -> None:
                    grad_tensors = []
                    param_tensors = []
                    for param in model.parameters():
                        if not param.requires_grad:
                            continue
                        param_tensors.append(param)
                        grad_tensors.append(
                            param.grad if param.grad is not None else torch.zeros_like(param)
                        )

                    if not grad_tensors:
                        return

                    grad_vector = torch.nn.utils.parameters_to_vector(grad_tensors)
                    direction_vector = correction_direction.to(
                        grad_vector.device,
                        dtype=grad_vector.dtype,
                    )
                    direction_norm_sq = torch.dot(direction_vector, direction_vector).clamp_min(1e-12)
                    parallel_component = torch.dot(grad_vector, direction_vector) / direction_norm_sq
                    projected_vector = grad_vector - parallel_component * direction_vector

                    offset = 0
                    for param in param_tensors:
                        param_size = param.numel()
                        projected_slice = projected_vector[offset:offset + param_size].view_as(param)
                        if param.grad is None:
                            param.grad = projected_slice.clone()
                        else:
                            param.grad.copy_(projected_slice)
                        offset += param_size

                total_loss.backward()
                # if diagnostics is not None:
                #     diagnostics.observe_batch(model, total_loss)
                # Constrained optimization:---------------------------------
                # _project_gradients_perpendicular_to_direction() 
                #--------------------------------------------------------------
                # optimizer.step()
                adam_optimizer.step()
                num_examples += labels.size(0)

                # # Constrained optimization -------------------------------
                update = model.get_weights() - pre_weights
                def _mask(pre_weights, sign_vector) -> None:
                    post_weights = model.get_weights()
                    update = post_weights - pre_weights
                    if sign_vector is None:
                        raise ValueError(
                            "perturbation_plus_correction requires gan_attack_payload.perturbation_sign"
                        )
                    sign_vector = sign_vector.to(update.device, dtype=update.dtype)
                    if sign_vector.shape != update.shape:
                        raise ValueError("perturbation_sign shape does not match model.get_weights()")
                    masked_update = torch.where(
                        update * sign_vector < 0,
                        torch.zeros_like(update),
                        update,
                    )
                    model.set_weights(pre_weights + masked_update)
                def _project_perpendicular(pre_weights, direction) -> None:
                    update = model.get_weights() - pre_weights
                    direction = direction.to(update.device, dtype=update.dtype)
                    if direction.shape != update.shape:
                        raise ValueError("projection direction shape does not match model.get_weights()")
                    projected_update = update - torch.dot(update, direction) / direction.dot(direction).clamp_min(1e-12) * direction
                    model.set_weights(pre_weights + projected_update)
                    # -------------------------------
                constrained_mode = correction_config.get("CONSTRAINED_MODE", "none")
                # Log and expose constrained mode for debugging/inspection
                # log(INFO, "GAN correction constrained_mode: %s", constrained_mode)
                if fitres is not None:
                    try:
                        fitres.metrics["constrained_mode"] = constrained_mode
                    except Exception:
                        # best-effort; don't break training if metrics dict is unexpected
                        pass
                if constrained_mode == "mask":
                    _mask(pre_weights, attack_payload.perturbation_sign)
                elif constrained_mode == "perpendicular":
                    _project_perpendicular(pre_weights, correction_direction)
                   
        # model.eval()
        
        #     if diagnostics is not None:
        #         diagnostics.end_epoch(model, epoch)

        # 

        # if diagnostics is not None:
        #     diagnostics.finalize(fitres)

        fitres.param_array["correction"] = model.get_weights()
    else:
        num_examples = len(trainloader.dataset) * max(int(epochs), 0)


    # Log L2 distances to the initial (pre-perturb) weights.
    if fitres is not None:
        pre_weights = fitres.param_array.get("pre")
        post_perturb = fitres.param_array.get("perturbation")
        post_corr = fitres.param_array.get("correction")
        if pre_weights is not None and post_perturb is not None:
            dist_perturb = torch.linalg.vector_norm(post_perturb - pre_weights).item()
            fitres.metrics["dist_perturb"] = float(dist_perturb)
        if pre_weights is not None and post_corr is not None:
            dist_corr = torch.linalg.vector_norm(post_corr - pre_weights).item()
            fitres.metrics["dist_corr"] = float(dist_corr)
        # Record the perturbation norm scale (``c``) so it travels with the filter
        # metrics into history and can be referenced in the detection-rate table.
        fitres.metrics["norm_scale"] = float(norm_scale)
        if pre_weights is not None:
            fitres.metrics["norm_pre"] = float(torch.linalg.vector_norm(pre_weights).item())
        if post_perturb is not None:
            fitres.metrics["norm_perturb"] = float(torch.linalg.vector_norm(post_perturb).item())
        if post_corr is not None:
            fitres.metrics["norm_corr"] = float(torch.linalg.vector_norm(post_corr).item())

    return num_examples, 0.0


def _flatten_grads(grads):
        return torch.cat([grad.reshape(-1) for grad in grads])


def compute_kd_loss(
        student_outputs: torch.Tensor,
        labels: torch.Tensor,
        criterion,
        teacher_outputs: Optional[torch.Tensor] = None,
        kd_alpha: float = 0.0,
        kd_temperature: float = 5.0,
        base_loss: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
    """Compute KD-combined loss with gradients flowing only through student outputs."""
    ce_loss = base_loss if base_loss is not None else criterion(student_outputs, labels)
    if teacher_outputs is None:
        return ce_loss

    kd_temperature = float(kd_temperature)
    kd_alpha = float(kd_alpha)
    kd_loss = F.kl_div(
        F.log_softmax(student_outputs / kd_temperature, dim=1),
        F.softmax(teacher_outputs.detach() / kd_temperature, dim=1),
        reduction="batchmean",
    ) * (kd_temperature ** 2)
    return (1.0 - kd_alpha) * ce_loss + kd_alpha * kd_loss


def _cosine_similarity_between_grads(grads_a, grads_b, eps: float = 1e-12) -> float:
        flat_a = _flatten_grads(grads_a)
        flat_b = _flatten_grads(grads_b)
        norm_a = torch.linalg.vector_norm(flat_a)
        norm_b = torch.linalg.vector_norm(flat_b)
        if norm_a.item() <= 0 or norm_b.item() <= 0:
            return 0.0
        cosine_similarity = torch.dot(flat_a, flat_b) / (norm_a * norm_b).clamp_min(eps)
        return float(cosine_similarity.item())