"""Training diagnostics for the perturbation+correction GAN attack.

These tools answer the questions that come up while debugging the
perturbation+correction attack: after we push the global model away by a fixed
magnitude, does training on synthetic data actually walk it *back* toward the
global optimum (which sits at a loss minimum by construction), and if not, why
-- is the synthetic-loss landscape flat, are the batch gradients
noisy/uninformative, are the updates sparse, or is the model just overfitting
the synthetic sample instead of generalising to fresh synthetic data drawn the
same way?

The module is intentionally decoupled from ``malicious_GAN``: the loss used for
the fresh-data evaluation is injected via ``loss_fn`` so there is no import
cycle.
"""

import json
import math
from logging import INFO
from typing import Dict, Optional

import torch
import torch.nn as nn

from fedml.common import FitRes, log


def _param_vector(model: nn.Module, *, grads: bool = False) -> torch.Tensor:
    """Flatten a model's parameters (or their gradients) into one vector.

    Uses the exact ordering of ``model.parameters()`` -- the same ordering as
    ``BaseModel.get_weights`` (``parameters_to_vector``) -- so the result aligns
    elementwise with ``model.get_weights()`` and with the perturbation /
    correction direction tensors. Buffers (e.g. BatchNorm running stats) are
    excluded, which is exactly what we want: gradients only exist for
    parameters, so the weight-space and gradient-space vectors stay consistent.
    """
    tensors = []
    for param in model.parameters():
        if grads:
            tensors.append(param.grad if param.grad is not None else torch.zeros_like(param))
        else:
            tensors.append(param)
    return nn.utils.parameters_to_vector(tensors).detach()


def _vector_density(vector: torch.Tensor, eps: float = 1e-12) -> float:
    """Hoyer-style density in ``(0, 1]``.

    ``||v||_1 / (sqrt(n) * ||v||_2)``. Equals 1.0 when every entry shares the
    same magnitude (fully dense) and tends to ``1/sqrt(n)`` when a single entry
    dominates (maximally sparse). A small value means the update/gradient is
    concentrated on a handful of coordinates.
    """
    n = vector.numel()
    if n == 0:
        return float("nan")
    l2 = torch.linalg.vector_norm(vector)
    if l2.item() <= eps:
        return 0.0
    l1 = vector.abs().sum()
    return float((l1 / (l2 * math.sqrt(n))).item())


def _fraction_near_zero(vector: torch.Tensor, rel_tol: float = 1e-3) -> float:
    """Fraction of entries whose magnitude is ``<= rel_tol * max|entry|``.

    A direct read on how sparse an update is: close to 1.0 means almost all
    coordinates barely move relative to the largest one.
    """
    n = vector.numel()
    if n == 0:
        return float("nan")
    max_abs = vector.abs().max()
    if max_abs.item() <= 0:
        return 1.0
    threshold = rel_tol * max_abs
    return float((vector.abs() <= threshold).float().mean().item())


def _safe_cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    """Cosine similarity that returns 0.0 for (near-)zero vectors."""
    norm_a = torch.linalg.vector_norm(a)
    norm_b = torch.linalg.vector_norm(b)
    if norm_a.item() <= eps or norm_b.item() <= eps:
        return 0.0
    return float((torch.dot(a, b) / (norm_a * norm_b)).item())


def _fmt(value) -> str:
    """Compact float formatting for log lines."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    if number != 0.0 and (abs(number) < 1e-3 or abs(number) >= 1e4):
        return f"{number:.3e}"
    return f"{number:.4f}"


class CorrectionDiagnostics:
    """Tracks task / loss-landscape / optimisation statistics during the
    synthetic-data correction phase of ``perturbation_plus_correction``.

    Instantiate *after* the model has been perturbed (so ``model`` is the
    perturbed starting point and ``global_model`` is the un-perturbed optimum),
    call :meth:`observe_batch` after every ``loss.backward()``, call
    :meth:`end_epoch` at the end of each epoch, and :meth:`finalize` once after
    training to push summary scalars into ``fitres.metrics``.

    ``loss_fn`` is the KD loss used during training (injected to avoid an import
    cycle); it is called as
    ``loss_fn(student_outputs=, labels=, criterion=, teacher_outputs=,
    kd_alpha=, kd_temperature=)`` so the fresh-data loss is directly comparable
    to the training loss.

    Per-epoch metrics, grouped by the question they answer:

    Is the model getting closer to the global model?
        ``dist_to_global``        L2 distance ``||theta - theta_global||``.
        ``recovery_fraction``     ``1 - dist_to_global / dist_start_to_global``;
                                  1.0 == back at the global optimum, 0.0 == no
                                  recovery, < 0 == drifted further than the
                                  initial perturbation. This is the "40 -> 24"
                                  number expressed as a fraction.
        ``delta_dist_to_global``  change since last epoch (< 0 == approaching).
        ``cos_descent_to_global`` cosine between the synthetic-loss descent
                                  direction (``-mean_grad``) and the direction
                                  home (``theta_global - theta``). The headline
                                  diagnostic: > 0 means descending the synthetic
                                  loss pulls toward the global optimum (recovery
                                  works); ~0 means the synthetic loss is
                                  orthogonal to / uninformative about the way
                                  home; < 0 means it actively pushes away.

    Are the batch gradients noisy / uninformative?
        ``grad_snr`` / ``grad_snr_db``  gradient signal-to-noise ratio,
                                  ``||E[g]||^2 / tr(Cov[g])`` over the epoch's
                                  batches. High == coherent, informative
                                  gradients; low == noise dominates.
        ``grad_coherence``        cosine between this epoch's mean gradient and
                                  the previous epoch's (temporal consistency).
        ``grad_norm``             ``||mean_grad||`` (tiny on a flat landscape).

    Are the updates extremely sparse?
        ``grad_density`` / ``update_density``          Hoyer density (see
                                  :func:`_vector_density`) of the mean gradient
                                  and of the net update from the perturbed start.
        ``grad_frac_near_zero`` / ``update_frac_near_zero``  fraction of nearly
                                  dead coordinates.

    Is it overfitting / does recovery generalise, and is the landscape flat?
        ``train_loss``            mean KD loss on the batches trained on.
        ``loss_delta``            change in ``train_loss`` vs last epoch (~0
                                  alongside a tiny ``grad_norm`` == flat region).
        ``fresh_loss`` / ``fresh_ce`` / ``fresh_acc``  same KD loss, plain
                                  cross-entropy and accuracy on a *freshly
                                  generated* synthetic batch (new noise + labels,
                                  same generator) that the model never trained
                                  on -- i.e. "other synthetic data, generated
                                  similarly".
        ``overfit_gap``           ``fresh_loss - train_loss`` (generalisation gap
                                  on fresh synthetic data).
    """

    def __init__(
        self,
        model: nn.Module,
        global_model: Optional[nn.Module],
        *,
        device,
        gen_model: nn.Module,
        criterion,
        loss_fn,
        num_classes: int,
        latent_size: int,
        kd_alpha: float,
        kd_temperature: float,
        fresh_size: int = 256,
        near_zero_rel_tol: float = 1e-3,
        client_id: str = "-",
    ) -> None:
        self.device = device
        self.global_model = global_model
        self.gen_model = gen_model
        self.criterion = criterion
        self.loss_fn = loss_fn
        self.num_classes = int(num_classes)
        self.latent_size = int(latent_size)
        self.kd_alpha = float(kd_alpha)
        self.kd_temperature = float(kd_temperature)
        self.fresh_size = int(fresh_size)
        self.near_zero_rel_tol = float(near_zero_rel_tol)
        self.client_id = client_id

        self.theta_start = _param_vector(model).to(device)  # perturbed starting point
        if global_model is not None:
            self.theta_global = _param_vector(global_model).to(device)
            self.dist_start_to_global = float(
                torch.linalg.vector_norm(self.theta_start - self.theta_global).item()
            )
        else:
            self.theta_global = None
            self.dist_start_to_global = float("nan")

        self._prev_mean_grad: Optional[torch.Tensor] = None
        self._prev_dist_to_global = self.dist_start_to_global
        self._prev_train_loss: Optional[float] = None
        self.history: list = []
        self._reset_epoch()

    def _reset_epoch(self) -> None:
        self._grad_sum: Optional[torch.Tensor] = None  # running sum of per-batch grads
        self._grad_sq_sum = 0.0                         # running sum of ||g_b||^2
        self._n_batches = 0
        self._loss_sum = 0.0
        self._loss_count = 0

    @torch.no_grad()
    def observe_batch(self, model: nn.Module, loss: torch.Tensor) -> None:
        """Accumulate the post-backward gradient and loss for one batch.

        Must be called after ``loss.backward()`` and before the gradients are
        zeroed; reads ``param.grad`` and does not modify training state.
        """
        grad = _param_vector(model, grads=True)
        if self._grad_sum is None:
            self._grad_sum = torch.zeros_like(grad)
        self._grad_sum += grad
        self._grad_sq_sum += float(torch.dot(grad, grad).item())
        self._n_batches += 1
        self._loss_sum += float(loss.detach().item())
        self._loss_count += 1

    @torch.no_grad()
    def _fresh_synthetic_eval(self, model: nn.Module) -> Dict[str, float]:
        """Evaluate on a freshly generated synthetic batch (no training)."""
        if self.fresh_size <= 0:
            return {}
        labels = torch.randint(0, self.num_classes, (self.fresh_size,), device=self.device)
        noise = torch.randn(self.fresh_size, self.latent_size, device=self.device)
        images = self.gen_model(noise, labels).detach()
        was_training = model.training
        model.eval()
        try:
            student_outputs = model(images)
            teacher_outputs = self.global_model(images) if self.global_model is not None else None
            fresh_loss = self.loss_fn(
                student_outputs=student_outputs,
                labels=labels,
                criterion=self.criterion,
                teacher_outputs=teacher_outputs,
                kd_alpha=self.kd_alpha,
                kd_temperature=self.kd_temperature,
            )
            fresh_ce = self.criterion(student_outputs, labels)
            fresh_acc = (student_outputs.argmax(dim=1) == labels).float().mean()
        finally:
            if was_training:
                model.train()
        return {
            "fresh_loss": float(fresh_loss.item()),
            "fresh_ce": float(fresh_ce.item()),
            "fresh_acc": float(fresh_acc.item()),
        }

    @torch.no_grad()
    def end_epoch(self, model: nn.Module, epoch: int) -> Optional[Dict[str, float]]:
        """Compute, log and store the diagnostic summary for one epoch."""
        if self._n_batches == 0:
            return None

        mean_grad = self._grad_sum / self._n_batches
        grad_norm = float(torch.linalg.vector_norm(mean_grad).item())
        expected_sq = self._grad_sq_sum / self._n_batches       # E[||g||^2]
        mean_sq = float(torch.dot(mean_grad, mean_grad).item())  # ||E[g]||^2
        grad_var = max(expected_sq - mean_sq, 0.0)              # tr(Cov[g])
        if grad_var > 1e-12:
            grad_snr = mean_sq / grad_var
        else:
            grad_snr = float("inf")
        grad_snr_db = (
            float(10.0 * math.log10(grad_snr))
            if grad_snr > 0.0 and math.isfinite(grad_snr)
            else float("nan")
        )

        theta = _param_vector(model).to(self.device)
        net_update = theta - self.theta_start
        dist_from_start = float(torch.linalg.vector_norm(net_update).item())
        train_loss = self._loss_sum / max(self._loss_count, 1)

        summary: Dict[str, float] = {
            "epoch": int(epoch),
            "grad_norm": grad_norm,
            "grad_snr": grad_snr,
            "grad_snr_db": grad_snr_db,
            "grad_density": _vector_density(mean_grad),
            "grad_frac_near_zero": _fraction_near_zero(mean_grad, self.near_zero_rel_tol),
            "update_density": _vector_density(net_update),
            "update_frac_near_zero": _fraction_near_zero(net_update, self.near_zero_rel_tol),
            "dist_from_start": dist_from_start,
            "train_loss": train_loss,
        }

        if self.theta_global is not None:
            to_global = self.theta_global - theta
            dist_to_global = float(torch.linalg.vector_norm(to_global).item())
            summary["dist_to_global"] = dist_to_global
            summary["delta_dist_to_global"] = dist_to_global - self._prev_dist_to_global
            summary["cos_descent_to_global"] = _safe_cosine(-mean_grad, to_global)
            if self.dist_start_to_global > 1e-12:
                summary["recovery_fraction"] = 1.0 - dist_to_global / self.dist_start_to_global
            self._prev_dist_to_global = dist_to_global

        summary["grad_coherence"] = (
            _safe_cosine(mean_grad, self._prev_mean_grad)
            if self._prev_mean_grad is not None
            else float("nan")
        )
        summary["loss_delta"] = (
            train_loss - self._prev_train_loss if self._prev_train_loss is not None else float("nan")
        )

        fresh = self._fresh_synthetic_eval(model)
        summary.update(fresh)
        if "fresh_loss" in fresh:
            summary["overfit_gap"] = fresh["fresh_loss"] - train_loss

        self.history.append(summary)
        log(
            INFO,
            "[diag][client %s][corr epoch %s] %s",
            self.client_id,
            epoch,
            self._format(summary),
        )

        self._prev_mean_grad = mean_grad
        self._prev_train_loss = train_loss
        self._reset_epoch()
        return summary

    def finalize(self, fitres: Optional[FitRes]) -> None:
        """Push last-epoch summary scalars into ``fitres.metrics`` (best effort)."""
        if fitres is None or not self.history:
            return
        last = self.history[-1]
        keys = (
            "dist_to_global", "recovery_fraction", "delta_dist_to_global",
            "cos_descent_to_global", "grad_norm", "grad_snr", "grad_snr_db",
            "grad_coherence", "grad_density", "update_density",
            "grad_frac_near_zero", "update_frac_near_zero", "dist_from_start",
            "train_loss", "loss_delta", "fresh_loss", "fresh_ce", "fresh_acc",
            "overfit_gap",
        )
        try:
            for key in keys:
                if key in last:
                    fitres.metrics[f"diag_{key}"] = float(last[key])
        except Exception:
            # best-effort; never break training because of diagnostics
            pass

    @staticmethod
    def _format(summary: Dict[str, float]) -> str:
        order = (
            "dist_to_global", "recovery_fraction", "delta_dist_to_global",
            "cos_descent_to_global", "grad_norm", "grad_snr_db", "grad_coherence",
            "grad_density", "update_density", "grad_frac_near_zero",
            "train_loss", "loss_delta", "fresh_loss", "fresh_acc", "overfit_gap",
        )
        return " ".join(f"{key}={_fmt(summary[key])}" for key in order if key in summary)
