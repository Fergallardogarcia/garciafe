"""Implementation of a weight-permutation (function-preserving) malicious client.

This attack exploits the *permutation symmetry* of neural networks. For any
hidden layer, permuting its output units and applying the inverse permutation to
the consuming layer's input dimension yields a network that computes *exactly the
same function* (cf. "Git Re-Basin", Ainsworth et al. 2022; FedMA, Wang et al.
2020). The local model's output distribution is therefore totally unchanged --
so any defense that inspects predictions, accuracy, or output statistics sees a
perfectly normal client.

What changes is the *coordinate representation* of the weights. FedAvg aggregates
parameters coordinate-wise, so averaging a permuted ("re-based") update with the
un-permuted honest updates mixes incompatible neuron orderings and corrupts the
aggregate -- the global model degrades even though every individual submission is
functionally valid. The more layers are permuted, the more thorough the
misalignment, so this client permutes *every* layer it can prove is
function-preserving.

Architecture support
--------------------
* Sequential feed-forward nets (LeNet, simple CNN/MLP): the leaf modules iterate
  in forward order, so consecutive parametric layers form producer/consumer
  pairs and every internal boundary is permuted.
* ResNets (custom ``resnet_custom`` and torchvision ``resnet_pytorch``): the
  residual stream constrains which permutations are legal. We permute
    - every block's *intermediate* channels (conv1 -> bn1 -> conv2), which are
      private to the block, with an independent permutation, and
    - the *residual-stream* channels of every stage consistently across the
      stem, all block conv outputs, batch-norms, downsample shortcuts and the
      classifier input. Identity shortcuts force input-perm == output-perm;
      downsample (conv) shortcuts let the stream be re-based between stages.
  Together these touch essentially every weight tensor in the network.

A built-in forward-pass check verifies (within float tolerance) that the
permuted model matches the honestly trained one before submission; if the
architecture has a structure this permuter cannot prove safe, it reverts to the
honest update rather than risk changing the output distribution.
"""

import copy
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from logging import DEBUG, WARNING
from fedml.common import (
    FitIns,
    FitRes,
    log,
)

from .honest_client import HonestClient

# Layers whose output units can be permuted (parametric "producers").
_PRODUCER_TYPES = (nn.Conv2d, nn.Linear)
# Normalization layers that ride along with a producer's output channels.
_NORM_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.InstanceNorm1d, nn.InstanceNorm2d)
# Normalization layers that grant a *scale* gauge freedom to the layer feeding
# them: scaling that layer by any non-zero c is absorbed by the (re-centering,
# re-scaling) norm, up to a sign that is compensated on the norm's affine gamma.
_GAUGE_NORM_TYPES = (
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
    nn.GroupNorm, nn.LayerNorm,
)


def parse_coordination(value):
    """Normalize the ``PERMUTER_CONFIG.COORDINATION`` field to one of:

        * ``None`` -- independent: every client draws a fresh permutation each
          round (seed = base + round + client_id).
        * ``int N`` (>=1) -- cyclic coordination: all clients share one
          permutation, held fixed for N-round cycles and recomputed each cycle.
        * ``"all"`` -- full coordination: one shared permutation for the whole run.

    Accepts YAML quirks (``none`` parses as the string "none", not Python None)
    and the legacy boolean field (``true`` -> "all", ``false`` -> None).
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return "all" if value else None
    if isinstance(value, int):
        return value if value >= 1 else None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("none", "null", "", "false", "independent"):
            return None
        if v in ("all", "true", "full"):
            return "all"
        try:
            n = int(v)
            return n if n >= 1 else None
        except ValueError:
            return None
    return None


class PermuterClient(HonestClient):
    """A malicious client that submits a function-preserving permutation.

    The client trains honestly, then randomly permutes the hidden units of as
    many layers as possible while exactly preserving the network's function. The
    submitted parameters look benign in output space but are mis-aligned in
    coordinate space, poisoning coordinate-wise FedAvg aggregation.

    Config (``attack_config["PERMUTER_CONFIG"]``, all optional)::

        SCALE_FACTOR : float  -- weights the submission in (weighted) FedAvg.
        TOLERANCE    : float  -- max allowed *relative* output deviation for the
                                 safety check (default 1e-2).
        SEED         : int    -- base seed; combined with round & client id so
                                 each round uses a fresh permutation.
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
        perm_cfg = (self.attack_config or {}).get("PERMUTER_CONFIG", {}) or {}
        self.scale_factor = perm_cfg.get("SCALE_FACTOR", 1.0)
        self.tolerance = perm_cfg.get("TOLERANCE", 1e-2)
        self.base_seed = perm_cfg.get("SEED", 0)
        # COORDINATION: None -> independent per client/round; int N -> shared
        # permutation held for N-round cycles; "all" -> one shared permutation for
        # the whole run. For both coordinated modes the seed is broadcast by the
        # server (gan_attack_prototype.broadcast_permutation_seed), so the client
        # just consumes the payload seed; only None reseeds per client/round.
        self.coordination = parse_coordination(perm_cfg.get("COORDINATION", None))
        # ---- BatchNorm/LayerNorm *scale* gauge symmetry ----
        # Layers feeding a normalization layer can be scaled by any non-zero c
        # (the norm absorbs it). GAUGE_SCALE_NORM_LAYERS is the magnitude |c| to
        # apply; GAUGE_SCALE_PROB is the per-layer APPLY probability (x ~ U[0,1];
        # x < prob -> apply, else skip), so scaling is non-uniform across layers
        # and clients. The sign of c is drawn +/- with fixed 0.5 probability.
        self.gauge_scale = perm_cfg.get("GAUGE_SCALE_NORM_LAYERS", 6.0)
        self.gauge_prob = perm_cfg.get("GAUGE_SCALE_PROB", 0.5)

    @property
    def client_type(self):
        """Returns current client's type."""
        return "PERMUTER"

    def fit(self, model, device, ins: FitIns) -> FitRes:
        # print(f"[Client {self.client_id}] fit, config: {ins.config}")

        # Don't perform attack until a specific round, and even then only with
        # the configured probability -- mirrors the other malicious clients.
        server_round = int(ins.config["server_round"])
        attack = np.random.random() < self.attack_config["ATTACK_RATIO"]

        if (server_round < self.attack_config["ATTACK_ROUND"]) or not attack:
            return super().fit(model, device, ins=ins)

        # Train honestly first -- the permutation is applied to the *result* of
        # ordinary local training so the contribution is otherwise indistinguishable.
        fit_results = super().fit(model, device, ins=ins)

        # Keep a pristine copy of the honest update to (a) run the safety check
        # against and (b) fall back to if the permutation cannot be proven safe.
        honest_parameters = fit_results.parameters.detach().clone()

        # Load the honest weights into the model and record the reference output.
        model.set_weights(honest_parameters, clone=True)
        model.to(device)
        ref_input = self._reference_batch(device)
        was_training = model.training
        model.eval()
        with torch.no_grad():
            ref_output = model(ref_input).detach().clone()

        # Apply random, function-preserving permutations to as many layers as the
        # architecture safely allows.
        if self.coordination is None:
            # Independent attack: a fresh permutation per client and per round.
            perm_seed = int(self.base_seed) + server_round * 100003 + int(self.client_id)
        else:
            # Coordinated attack ("all" or int N-round cycles): use the shared seed
            # broadcast by the server, which already encodes the cycle logic, so all
            # malicious clients compute the SAME permutation. Fall back to the base
            # seed if no payload was attached.
            payload = getattr(ins, "gan_attack_payload", None)
            payload_seed = getattr(payload, "permutation_seed", None) if payload is not None else None
            perm_seed = int(payload_seed) if payload_seed is not None else int(self.base_seed)
        generator = torch.Generator(device=device)
        generator.manual_seed(perm_seed)
        num_permuted = _apply_function_preserving_permutations(model, generator, device)

        # Stack the BatchNorm/LayerNorm scale gauge on top of the permutation:
        # randomly (per layer, per client) scale layers feeding a norm by
        # +/-GAUGE_SCALE_NORM_LAYERS, which the norm absorbs -- another way to move
        # the weights far in coordinate space while leaving the function intact.
        num_gauged = _apply_gauge_scaling(
            model, generator, self.gauge_scale, self.gauge_prob, device
        )

        # Verify the function is genuinely unchanged before we submit anything.
        with torch.no_grad():
            new_output = model(ref_input).detach()
        scale = ref_output.abs().max().item() if ref_output.numel() else 0.0
        max_dev = (new_output - ref_output).abs().max().item() if ref_output.numel() else 0.0
        rel_dev = max_dev / (scale + 1e-8)

        if was_training:
            model.train()

        if (num_permuted + num_gauged) == 0 or rel_dev > self.tolerance:
            # Could not safely transform this architecture or the check failed --
            # revert to the honest update untouched.
            log(
                WARNING,
                "[PermuterClient %s] transform aborted (perm=%d, gauge=%d, rel_dev=%.2e > tol=%.2e); "
                "submitting honest update.",
                self.client_id, num_permuted, num_gauged, rel_dev, self.tolerance,
            )
            model.set_weights(honest_parameters, clone=True)
            del fit_results.parameters
            fit_results.parameters = honest_parameters
            fit_results.metrics["attacking"] = False
            fit_results.metrics["permuted_layers"] = 0
            fit_results.metrics["gauge_scaled_layers"] = 0
            return fit_results

        # Read the re-based parameters back out and submit them.
        permuted_parameters = model.get_weights()
        if self._process:
            permuted_parameters = permuted_parameters.cpu()

        del fit_results.parameters
        fit_results.parameters = permuted_parameters
        fit_results.num_examples *= self.scale_factor
        fit_results.metrics["attacking"] = True
        fit_results.metrics["permuted_layers"] = num_permuted
        fit_results.metrics["gauge_scaled_layers"] = num_gauged

        log(
            DEBUG,
            "[PermuterClient %s] round %d: permuted %d layers, gauge-scaled %d layers "
            "(output unchanged, rel_dev=%.2e).",
            self.client_id, server_round, num_permuted, num_gauged, rel_dev,
        )

        return fit_results

    def _reference_batch(self, device, max_samples: int = 64):
        """A small input batch used to verify the function is preserved.

        Any fixed input works -- we only compare the model's output on the *same*
        input before and after permuting -- so we use raw training samples.
        """
        data = self._trainset.data[:max_samples].detach().clone()
        return data.to(device=device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Permutation helpers
# ---------------------------------------------------------------------------

def _collect_norm_scalable_layers(model):
    """Return ``(layer, norm)`` pairs where ``layer`` (Conv/Linear) feeds directly
    into a normalization ``norm`` -- i.e. the layers that carry a scale gauge.

    A producer "feeds directly into" a norm when the norm is the very next leaf
    module in forward order (Conv -> BN with nothing in between), which is the
    only case where uniformly scaling the producer is exactly absorbed.
    """
    leaves = [m for m in model.modules() if len(list(m.children())) == 0]
    pairs = []
    for idx, module in enumerate(leaves):
        if isinstance(module, _PRODUCER_TYPES):
            if idx + 1 < len(leaves) and isinstance(leaves[idx + 1], _GAUGE_NORM_TYPES):
                pairs.append((module, leaves[idx + 1]))
    return pairs


def _apply_gauge_scaling(model, generator, scale, prob, device) -> int:
    """Apply the BatchNorm/LayerNorm *scale* gauge to a random subset of layers.

    For each (layer -> norm) pair: draw x ~ U[0,1]; apply when ``x < prob``,
    otherwise skip. When applied, scale the layer's weight/bias by
    ``c = sign * |scale|`` (sign drawn
    +/- at p=0.5). The norm absorbs ``|c|``; only the sign is compensated, by
    flipping the norm's affine gamma. Running statistics (when tracked) are scaled
    by ``c`` (mean) and ``c**2`` (var). Returns the number of layers scaled.

    The transform is exactly function-preserving for affine norms; for a norm
    without an affine weight there is no gamma to flip, so only positive scales
    are used there.
    """
    if scale is None or float(scale) == 0.0:
        return 0
    magnitude = abs(float(scale))
    pairs = _collect_norm_scalable_layers(model)
    count = 0
    with torch.no_grad():
        for layer, norm in pairs:
            x = torch.rand((), generator=generator, device=device).item()
            if x >= prob:
                continue  # apply only when x < prob (GAUGE_SCALE_PROB = apply prob)

            has_affine = getattr(norm, "weight", None) is not None
            sign_draw = torch.rand((), generator=generator, device=device).item()
            # Negative scale needs an affine gamma to flip; fall back to positive
            # when the norm has no affine weight so the function stays preserved.
            sign = -1.0 if (sign_draw < 0.5 and has_affine) else 1.0
            c = sign * magnitude

            layer.weight.data.mul_(c)
            if getattr(layer, "bias", None) is not None:
                layer.bias.data.mul_(c)

            if has_affine:
                norm.weight.data.mul_(sign)
            if getattr(norm, "running_mean", None) is not None:
                norm.running_mean.data.mul_(c)
            if getattr(norm, "running_var", None) is not None:
                norm.running_var.data.mul_(c * c)
            count += 1
    return count


def _apply_function_preserving_permutations(model, generator, device) -> int:
    """Permute as many layers as possible in place; return the count applied."""
    if _looks_like_resnet(model):
        return _permute_resnet(model, generator, device)
    return _permute_sequential_chain(model, generator, device)


def _perm_norm(norm, perm) -> None:
    """Reorder a normalization layer's per-channel tensors by ``perm`` in place."""
    if norm is None:
        return
    for attr in ("weight", "bias", "running_mean", "running_var"):
        t = getattr(norm, attr, None)
        if t is not None:
            t.data = t.data[perm].contiguous()


def _perm_conv_or_linear(layer, out_perm=None, in_perm=None) -> None:
    """Reorder a Conv2d/Linear weight (and bias) along output and/or input dims.

    ``in_perm`` may be a 1-D index tensor matching the input dimension, *or* a
    ``(channel_perm, block)`` pair for a Conv->flatten->Linear boundary where
    each input channel maps to a contiguous block of ``block`` columns.
    """
    w = layer.weight.data
    if out_perm is not None:
        w = w[out_perm]
        if getattr(layer, "bias", None) is not None:
            layer.bias.data = layer.bias.data[out_perm].contiguous()
    if in_perm is not None:
        if isinstance(in_perm, tuple):
            channel_perm, block = in_perm
            out_features = w.shape[0]
            n = channel_perm.shape[0]
            w = w.view(out_features, n, block)[:, channel_perm, :].reshape(out_features, -1)
        else:
            w = w[:, in_perm]
    layer.weight.data = w.contiguous()


# --------------------------- sequential networks ---------------------------

def _collect_producer_chain(model):
    """Ordered permutable layers with their trailing norms for sequential nets."""
    leaves = [m for m in model.modules() if len(list(m.children())) == 0]
    chain = []
    for idx, module in enumerate(leaves):
        if isinstance(module, _PRODUCER_TYPES):
            norm = None
            if idx + 1 < len(leaves) and isinstance(leaves[idx + 1], _NORM_TYPES):
                norm = leaves[idx + 1]
            chain.append({"layer": module, "norm": norm})
    return chain


def _permute_sequential_chain(model, generator, device) -> int:
    """Permute every internal boundary of a feed-forward chain. Returns count.

    The final layer's outputs are never permuted (that would reorder the class
    logits and change the output distribution).
    """
    chain = _collect_producer_chain(model)
    count = 0
    with torch.no_grad():
        for producer, consumer in zip(chain[:-1], chain[1:]):
            p_layer = producer["layer"]
            c_layer = consumer["layer"]
            n = p_layer.weight.shape[0]

            # Work out how the consumer reads the producer's n output units.
            if isinstance(c_layer, nn.Linear):
                in_features = c_layer.weight.shape[1]
                if isinstance(p_layer, nn.Conv2d):
                    if in_features % n != 0:
                        continue
                    block = in_features // n
                else:  # Linear -> Linear
                    if in_features != n:
                        continue
                    block = 1
            elif isinstance(c_layer, nn.Conv2d):
                if c_layer.weight.shape[1] != n:
                    continue
                block = None
            else:
                continue

            perm = torch.randperm(n, generator=generator, device=device)

            _perm_conv_or_linear(p_layer, out_perm=perm)
            _perm_norm(producer["norm"], perm)

            if isinstance(c_layer, nn.Linear) and isinstance(p_layer, nn.Conv2d):
                _perm_conv_or_linear(c_layer, in_perm=(perm, block))
            else:
                _perm_conv_or_linear(c_layer, in_perm=perm)
            count += 1
    return count


# ------------------------------- resnets -----------------------------------

def _looks_like_resnet(model) -> bool:
    return all(hasattr(model, f"layer{i}") for i in (1, 2, 3, 4))


def _block_shortcut_conv_bn(block):
    """Return ``(conv, bn)`` of a block's projection shortcut, or ``(None, None)``.

    Handles both the custom ResNet (``block.shortcut`` Sequential, empty when the
    shortcut is identity) and torchvision (``block.downsample`` = None or
    Sequential).
    """
    sc = getattr(block, "shortcut", None)
    if sc is None:
        sc = getattr(block, "downsample", None)
    if sc is None:
        return None, None
    convs = [m for m in sc.modules() if isinstance(m, nn.Conv2d)]
    bns = [m for m in sc.modules() if isinstance(m, _NORM_TYPES)]
    if not convs:
        return None, None
    return convs[0], (bns[0] if bns else None)


def _permute_block(block, r_in, r_out, generator, device) -> int:
    """Permute one residual block in place. Returns number of convs touched.

    ``r_in`` / ``r_out`` are the input/output residual-stream permutations the
    block must respect; intermediate channels are permuted independently.
    """
    rand = lambda n: torch.randperm(n, generator=generator, device=device)
    count = 0

    # Ordered (conv, bn) pairs inside the block's main path.
    convs = [getattr(block, name) for name in ("conv1", "conv2", "conv3")
             if hasattr(block, name)]
    bns = [getattr(block, name) for name in ("bn1", "bn2", "bn3")
           if hasattr(block, name)]

    prev_perm = r_in
    for i, conv in enumerate(convs):
        is_last = (i == len(convs) - 1)
        # Last conv writes the residual stream (r_out); others use a fresh
        # intermediate permutation private to the block.
        out_perm = r_out if is_last else rand(conv.weight.shape[0])
        _perm_conv_or_linear(conv, out_perm=out_perm, in_perm=prev_perm)
        if i < len(bns):
            _perm_norm(bns[i], out_perm)
        prev_perm = out_perm
        count += 1

    # Projection shortcut (if any) must map r_in -> r_out.
    sc_conv, sc_bn = _block_shortcut_conv_bn(block)
    if sc_conv is not None:
        _perm_conv_or_linear(sc_conv, out_perm=r_out, in_perm=r_in)
        _perm_norm(sc_bn, r_out)
        count += 1

    return count


def _permute_resnet(model, generator, device) -> int:
    """Permute a ResNet end-to-end while preserving the residual stream."""
    rand = lambda n: torch.randperm(n, generator=generator, device=device)
    count = 0

    with torch.no_grad():
        # --- stem: conv1 -> bn1 ---
        stem_conv = model.conv1
        stem_bn = getattr(model, "bn1", None)
        p_prev = rand(stem_conv.weight.shape[0])
        # Input channels are the raw image; only permute the stem's outputs.
        _perm_conv_or_linear(stem_conv, out_perm=p_prev)
        _perm_norm(stem_bn, p_prev)
        count += 1

        # --- stages ---
        stages = [getattr(model, f"layer{i}") for i in (1, 2, 3, 4)
                  if hasattr(model, f"layer{i}")]
        for stage in stages:
            blocks = [b for b in stage.children()]
            if not blocks:
                continue
            block0 = blocks[0]
            sc_conv, _ = _block_shortcut_conv_bn(block0)
            if sc_conv is not None:
                # Projection shortcut: the stream may be re-based for this stage.
                p_stage = rand(sc_conv.weight.shape[0])
            else:
                # Identity shortcut: input and output streams must coincide.
                p_stage = p_prev

            for bi, block in enumerate(blocks):
                r_in = p_prev if bi == 0 else p_stage
                count += _permute_block(block, r_in, p_stage, generator, device)

            p_prev = p_stage

        # --- classifier: permute its input to match the final residual stream ---
        classifier = getattr(model, "linear", None)
        if classifier is None:
            classifier = getattr(model, "fc", None)
        if isinstance(classifier, nn.Linear):
            in_features = classifier.weight.shape[1]
            n = p_prev.shape[0]
            if in_features == n:
                _perm_conv_or_linear(classifier, in_perm=p_prev)
                count += 1
            elif in_features % n == 0:
                # Spatial dims survive pooling: each channel is a column block.
                _perm_conv_or_linear(classifier, in_perm=(p_prev, in_features // n))
                count += 1

    return count
