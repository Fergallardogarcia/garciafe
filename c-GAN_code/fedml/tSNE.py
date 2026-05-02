"""Module to run a single experiment on slurm like environment."""

import warnings
warnings.filterwarnings("ignore")

import multiprocessing
from logging import DEBUG
from typing import cast, Optional, Any

import copy
import torch
import os
from os.path import join
import argparse
import ntpath

import fedml
from fedml.common import log

from fedml.client import create_client
from fedml.configs import parse_configs
from fedml.data_handler import load_and_fetch_split, merge_splits
from fedml.models import load_model
from fedml.modules import ExperimentManager, setup_random_seeds
from fedml.server import (
    create_server,
    get_client_manager
)
from fedml.strategy import get_strategy
from fedml.defenses.filters.gan_filter import generate_dataset

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from sklearn.decomposition import PCA
import torch.nn.functional as F
try:
    import plotly.graph_objects as go
except ImportError:
    go = None

from pathlib import Path




def tSNE(
        exp_name,
        user_configs,
        executor_type,
        num_gpus=None,
        run_results_path: Optional[str] = None,
        max_samples_per_source: Optional[int] = None,
    ):

    def make_rng(seed: int) -> Any:
        default_rng_fn = getattr(np.random, "default_rng", None)
        if callable(default_rng_fn):
            return default_rng_fn(seed)
        return np.random.RandomState(seed)

    # Extract required user configurations
    total_clients = user_configs["SERVER_CONFIGS"]["MIN_NUM_CLIENTS"]

    # Get run device information
    run_devices = None
    if (user_configs["CLIENT_CONFIGS"]["RUN_DEVICE"] == "auto") and (num_gpus is not None):
        # run_devices = [f"cuda:{i%num_gpus}" for i in range(min_sample_size)]
        run_devices = [f"cuda:{i%num_gpus}" for i in range(num_gpus)]
    else:
        # run_devices = [user_configs["CLIENT_CONFIGS"]["RUN_DEVICE"] for i in range(min_sample_size)]
        run_devices = [user_configs["CLIENT_CONFIGS"]["RUN_DEVICE"]]

    log(DEBUG, f"Run device: {run_devices[0]}")

    # Load all dataset and make splits 
    (train_splits, split_labels), testset = load_and_fetch_split(n_clients=total_clients, dataset_conf=user_configs["DATASET_CONFIGS"])

    # Load appropriate number of local models
    
    if run_results_path is None:
        run_results_path = user_configs["OUTPUT_CONFIGS"]["RESULT_LOG_PATH"]
    folder = Path(str(run_results_path))

    # Only files directly inside the folder
    file_paths = sorted(p for p in folder.iterdir() if p.is_file())

    # Keep only generator weight files saved at the end of training.
    gen_model_paths = [
        p for p in file_paths
        if p.suffix == ".pt" and ("gen-def-weights-" in p.name or "gen-attack-weights-" in p.name) and ("TEST1" in p.name) # or "gen-attack-weights-" in p.name
    ]

    discriminator_model_paths = [
        p for p in file_paths   if p.suffix == ".pt" and ("weights-global-pre-attack-round-24" in p.name) #  "weights-global-pre-attack-round-24" , "TEST_DCGAN"
    ]

    def infer_gen_model_name(file_name: str):
        file_name = file_name.upper()
        if "DCGAN" in file_name:
            return "GEN-DCGAN"
        if "SIGMOID" in file_name:
            return "TEST-SIGMOID"
        if "TANH" in file_name:
            return "TEST-TANH"
        return None


    mod_conf= {"MODEL_NAME": "TEST-TANH", "NUM_CLASSES": 10, "LATENT_SIZE": 100, "OUT_CHANNEL": 3, "OUTPUT_SIZE": 32}
    MODls_configs=[]
    gen_models= []
    for path in gen_model_paths:
        mod_name = infer_gen_model_name(path.name)
        if mod_name is None:
            continue

        gen_model_conf = {**mod_conf, "MODEL_NAME": mod_name, "WEIGHT_PATH": str(path)}
        MODls_configs.append(gen_model_conf)

        gen_model = load_model(model_configs=gen_model_conf)
        saved_obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(saved_obj, dict):
            gen_model.load_state_dict(saved_obj)
        else:
            gen_model.set_weights(saved_obj)
        gen_model.eval()
        gen_models.append(gen_model)

    model_conf = {**mod_conf, "MODEL_NAME": "RESNET-18-CUSTOM", "WEIGHT_PATH": None}
    dis_model = load_model(model_configs=model_conf)
    saved_obj = torch.load(discriminator_model_paths[0], map_location="cpu", weights_only=False)
    if isinstance(saved_obj, dict):
        dis_model.load_state_dict(saved_obj)
    else:
        dis_model.set_weights(saved_obj)

    # Build one combined real dataset from all splits, then generate with each model.
    if len(train_splits) == 1:
        real_split = train_splits[0]
    else:
        real_split = merge_splits(train_splits)

    if hasattr(real_split, "targets"):
        input_classes = torch.as_tensor(getattr(real_split, "targets")).detach().clone().cpu().long()
    else:
        input_classes = torch.tensor(
            [int(real_split[idx][1]) for idx in range(len(real_split))],
            dtype=torch.long,
        )
    input_znoises = torch.randn(input_classes.size(0), mod_conf["LATENT_SIZE"])

    generated_datasets = []
    for model_conf, gen_model in zip(MODls_configs, gen_models):
        gen_dataset = generate_dataset(
            gen_model=gen_model,
            input_znoises=input_znoises,
            input_classes=input_classes,
            device="cpu",
            batch_size=1024,
        )
        generated_datasets.append((model_conf["MODEL_NAME"], gen_dataset))
        log(DEBUG, f"Generated {len(gen_dataset)} samples using {model_conf['MODEL_NAME']}")

    if len(generated_datasets) == 0:
        log(DEBUG, "No generator datasets found to run t-SNE")
        return

    # Build last-layer feature representations over real split0 plus generated datasets.
    

    def extract_xy(dataset):
        if hasattr(dataset, "tensors"):
            x, y = dataset.tensors
            return x.detach().cpu().float(), y.detach().cpu().long()
        if hasattr(dataset, "data") and hasattr(dataset, "targets"):
            x = torch.as_tensor(getattr(dataset, "data")).detach().cpu().float()
            y = torch.as_tensor(getattr(dataset, "targets")).detach().cpu().long()
            return x, y
        x_items = []
        y_items = []
        for idx in range(len(dataset)):
            x_i, y_i = dataset[idx]
            x_items.append(torch.as_tensor(x_i).detach().cpu().float())
            y_items.append(int(y_i))
        return torch.stack(x_items), torch.tensor(y_items, dtype=torch.long)

    def _to_nchw_float01(x: torch.Tensor) -> torch.Tensor:
        x = x.detach().cpu()
        if x.dim() == 4 and x.shape[-1] in (1, 3) and x.shape[1] not in (1, 3):
            x = x.permute(0, 3, 1, 2)
        x = x.float()
        if x.numel() > 0 and x.max().item() > 1.0:
            x = x / 255.0
        return x

    def extract_last_layer_repr(model, data_x: torch.Tensor, batch_size: int = 1024, layer: int = 3) -> torch.Tensor:
        model = model.to("cpu")
        model.eval()
        reps = []
        with torch.no_grad():
            for start in range(0, data_x.size(0), batch_size):
                x_batch = data_x[start:start + batch_size]
                if hasattr(model, "features"):
                    rep_batch = model.features(x_batch, output_layer=layer)
                elif hasattr(model, "forward_embedding"):
                    rep_batch = model.forward_embedding(x_batch)
                else:
                    rep_batch = model(x_batch)
                reps.append(rep_batch.detach().cpu())
        return torch.cat(reps, dim=0)

    def extract_classifier_repr(model, data_x: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
        """Extract embeddings aligned with discriminator class-weight vectors."""
        model = model.to("cpu")
        model.eval()
        reps = []
        with torch.no_grad():
            for start in range(0, data_x.size(0), batch_size):
                x_batch = data_x[start:start + batch_size]
                if hasattr(model, "forward_embedding"):
                    rep_batch = model.forward_embedding(x_batch)
                elif hasattr(model, "features"):
                    rep_batch = model.features(x_batch, output_layer=5)
                else:
                    rep_batch = model(x_batch)
                reps.append(rep_batch.detach().cpu().reshape(rep_batch.size(0), -1))
        return torch.cat(reps, dim=0)

    def extract_logits(model, data_x: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
        """Extract discriminator logits for per-class accuracy/loss statistics."""
        model = model.to("cpu")
        model.eval()
        logits = []
        with torch.no_grad():
            for start in range(0, data_x.size(0), batch_size):
                x_batch = data_x[start:start + batch_size]
                logits.append(model(x_batch).detach().cpu())
        return torch.cat(logits, dim=0)

    real_sources = [("real", real_split)]
    if len(train_splits) == 1:
        log(DEBUG, "Only one real split available; continuing with split0 only")

    synthetic_sources = [(name, ds) for name, ds in generated_datasets]
    source_names = [name for name, _ in (real_sources + synthetic_sources)]
    all_datasets = [ds for _, ds in (real_sources + synthetic_sources)]

    if max_samples_per_source is not None and max_samples_per_source <= 0:
        max_samples_per_source = None

    sampling_rng = make_rng(seed=user_configs["SERVER_CONFIGS"]["RANDOM_SEED"])

    real_x = []
    all_x = []
    all_y = []
    all_source_ids = []
    source_classifier_repr_tensors = []
    source_label_tensors = []
    source_logits_tensors = []
    for source_id, dataset in enumerate(all_datasets):
        data_x, data_y = extract_xy(dataset)

        if max_samples_per_source is not None and data_y.size(0) > max_samples_per_source:
            selected_idx = sampling_rng.choice(data_y.size(0), size=max_samples_per_source, replace=False)
            selected_idx = torch.as_tensor(selected_idx, dtype=torch.long, device="cpu")
            data_x = data_x[selected_idx]
            data_y = data_y[selected_idx]

        data_x = _to_nchw_float01(data_x)
        data_repr = extract_last_layer_repr(model=dis_model, data_x=data_x, batch_size=1024)
        classifier_repr = extract_classifier_repr(model=dis_model, data_x=data_x, batch_size=1024)
        logits = extract_logits(model=dis_model, data_x=data_x, batch_size=1024)

        if source_id < len(real_sources):
            real_x.append(data_repr.numpy())
        all_x.append(data_repr.numpy())
        all_y.append(data_y.numpy())
        all_source_ids.append(np.full(data_y.size(0), source_id, dtype=int))
        source_classifier_repr_tensors.append(classifier_repr)
        source_label_tensors.append(data_y.detach().cpu().long())
        source_logits_tensors.append(logits)

    real_features = np.concatenate(real_x, axis=0)
    features = np.concatenate(all_x, axis=0)
    labels = np.concatenate(all_y, axis=0)
    source_ids = np.concatenate(all_source_ids, axis=0)

    if real_features.shape[0] < 3 or real_features.shape[1] < 3:
        raise ValueError(
            f"Need at least 3 real samples and 3 feature dimensions for PCA(n_components=3), got {real_features.shape}"
        )

    pca = PCA(
        n_components=3,
        random_state=user_configs["SERVER_CONFIGS"]["RANDOM_SEED"],
    )
    pca.fit(real_features)
    embedding = pca.transform(features)

    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    cmap = cm.get_cmap("tab10")
    synthetic_colors = [
        "#ff1744",
        "#00e5ff",
        "#ffea00",
        "#76ff03",
        "#d500f9",
        "#ff9100",
        "#1de9b6",
        "#f50057",
    ]

    fig = plt.figure(figsize=(11, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    for source_id, source_name in enumerate(source_names):
        mask = source_ids == source_id
        if source_id < len(real_sources):
            point_colors = labels[mask]
            point_cmap = cmap
            point_size = 4
            point_alpha = 0.35
        else:
            synthetic_index = source_id - len(real_sources)
            point_colors = synthetic_colors[synthetic_index % len(synthetic_colors)]
            point_cmap = None
            point_size = 8
            point_alpha = 0.75
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            embedding[mask, 2],
            c=point_colors,
            cmap=point_cmap,
            vmin=0,
            vmax=9,
            marker=MarkerStyle(markers[source_id % len(markers)]),
            s=point_size,
            alpha=point_alpha,
            linewidths=0.0,
        )

    ax.set_title("PCA 3D (fit on real: merged splits; projected synthetic)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    source_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[i % len(markers)],
            linestyle="",
            markerfacecolor=("black" if i < len(real_sources) else synthetic_colors[(i - len(real_sources)) % len(synthetic_colors)]),
            markeredgecolor=("black" if i < len(real_sources) else synthetic_colors[(i - len(real_sources)) % len(synthetic_colors)]),
            markersize=6,
            label=name,
        )
        for i, name in enumerate(source_names)
    ]
    source_legend = ax.legend(handles=source_handles, title="Source", loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.add_artist(source_legend)

    class_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=cmap(class_id / 9.0),
            markeredgecolor=cmap(class_id / 9.0),
            markersize=6,
            label=f"Class {class_id}",
        )
        for class_id in range(10)
    ]
    ax.legend(handles=class_handles, title="Class", loc="lower left", bbox_to_anchor=(1.01, 0.0))

    out_dir = Path(user_configs["OUTPUT_CONFIGS"]["RESULT_LOG_PATH"])
    out_plot = out_dir / f"pca3_lastlayer_realfit_projected_{exp_name}.png"
    out_npz = out_dir / f"pca3_lastlayer_realfit_projected_{exp_name}.npz"
    out_html = out_dir / f"pca3_lastlayer_realfit_projected_{exp_name}.html"
    out_cos_plot = out_dir / f"class_alignment_cosine_norm_{exp_name}.png"
    out_cos_npz = out_dir / f"class_alignment_cosine_norm_{exp_name}.npz"
    out_cls_plot = out_dir / f"class_accuracy_crossentropy_{exp_name}.png"
    out_cls_npz = out_dir / f"class_accuracy_crossentropy_{exp_name}.npz"
    out_dist_plot = out_dir / f"class_mean_distance_{exp_name}.png"
    out_dist_npz = out_dir / f"class_mean_distance_{exp_name}.npz"
    fig.savefig(str(out_plot), dpi=220, bbox_inches="tight")
    np.savez(
        out_npz,
        embedding=embedding,
        labels=labels,
        source_ids=source_ids,
        source_names=np.array(source_names, dtype=object),
    )

    if not hasattr(dis_model, "linear") or not hasattr(dis_model.linear, "weight"):
        raise ValueError("Discriminator model must expose linear.weight to compute class-vector alignment")

    class_weight_vectors = dis_model.linear.weight.detach().cpu().float()
    class_weight_norms = torch.linalg.norm(class_weight_vectors, dim=1).numpy()
    n_classes = class_weight_vectors.size(0)
    class_ids = np.arange(n_classes)

    real_colors = ["#444444", "#7a7a7a", "#9a9a9a", "#b0b0b0"]
    metric_source_names = source_names
    cosine_rows = []
    for source_id, source_name in enumerate(metric_source_names):
        source_labels = source_label_tensors[source_id]
        classifier_repr = source_classifier_repr_tensors[source_id]

        if classifier_repr.size(0) != source_labels.size(0):
            raise ValueError(
                f"Representation/label size mismatch for {source_name}: "
                f"{classifier_repr.size(0)} vs {source_labels.size(0)}"
            )

        if classifier_repr.size(1) != class_weight_vectors.size(1):
            raise ValueError(
                f"Embedding dim mismatch for {source_name}: "
                f"features={classifier_repr.size(1)} vs class_weight={class_weight_vectors.size(1)}"
            )

        row = np.full(n_classes, np.nan, dtype=np.float32)
        for class_id in range(n_classes):
            class_mask = source_labels == class_id
            if torch.any(class_mask):
                class_mean = classifier_repr[class_mask].mean(dim=0)
                cosine_val = F.cosine_similarity(
                    class_mean.unsqueeze(0),
                    class_weight_vectors[class_id].unsqueeze(0),
                    dim=1,
                ).item()
                row[class_id] = cosine_val
        cosine_rows.append(row)

    cosine_by_source = np.vstack(cosine_rows) if cosine_rows else np.empty((0, n_classes), dtype=np.float32)
    cosine_std_by_source = np.nanstd(cosine_by_source, axis=1) if cosine_by_source.size else np.empty((0,), dtype=np.float32)

    projection_mean_rows = []
    projection_std_rows = []
    for source_id, source_name in enumerate(metric_source_names):
        source_labels = source_label_tensors[source_id]
        classifier_repr = source_classifier_repr_tensors[source_id]

        proj_mean_row = np.full(n_classes, np.nan, dtype=np.float32)
        proj_std_row = np.full(n_classes, np.nan, dtype=np.float32)
        for class_id in range(n_classes):
            class_mask = source_labels == class_id
            if not torch.any(class_mask):
                continue
            weight_vec = class_weight_vectors[class_id]
            weight_norm = torch.linalg.norm(weight_vec)
            if weight_norm <= 0:
                continue
            unit_weight_vec = weight_vec / weight_norm
            class_repr = classifier_repr[class_mask]
            projections = torch.abs(class_repr @ unit_weight_vec)
            proj_mean_row[class_id] = projections.mean().item()
            proj_std_row[class_id] = projections.std(unbiased=False).item()

        projection_mean_rows.append(proj_mean_row)
        projection_std_rows.append(proj_std_row)

    projection_mean_by_source = (
        np.vstack(projection_mean_rows) if projection_mean_rows else np.empty((0, n_classes), dtype=np.float32)
    )
    projection_std_by_source = (
        np.vstack(projection_std_rows) if projection_std_rows else np.empty((0, n_classes), dtype=np.float32)
    )

    fig_align, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    ax_cos, ax_norm = axes

    for class_id in class_ids:
        ax_cos.axvline(class_id, color="#d9d9d9", linewidth=0.7, zorder=0)

    for idx, source_name in enumerate(metric_source_names):
        if idx < len(real_sources):
            color = real_colors[idx % len(real_colors)]
        else:
            color = synthetic_colors[(idx - len(real_sources)) % len(synthetic_colors)]

        y_values = cosine_by_source[idx]
        valid_mask = ~np.isnan(y_values)
        if not np.any(valid_mask):
            continue

        x_values = class_ids[valid_mask]
        y_values = y_values[valid_mask]
        y_err = np.full(y_values.shape, cosine_std_by_source[idx], dtype=np.float32)
        ax_cos.errorbar(
            x_values,
            y_values,
            yerr=y_err,
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=color,
            label=source_name,
            capsize=3,
        )

    ax_cos.set_ylabel("Cosine Similarity")
    ax_cos.set_ylim(-1.05, 1.05)
    ax_cos.set_title("Class-wise alignment: class weight vs class mean embedding")
    if len(metric_source_names) > 0:
        ax_cos.legend(title="Source", loc="best")

    for class_id in class_ids:
        ax_norm.axvline(class_id, color="#e6e6e6", linewidth=0.7, zorder=0)
    ax_norm.plot(class_ids, class_weight_norms, marker="s", color="#1f77b4", linewidth=1.8, label="||w_c||")
    ax_proj = ax_norm.twinx()

    for idx, source_name in enumerate(metric_source_names):
        if idx < len(real_sources):
            color = real_colors[idx % len(real_colors)]
        else:
            color = synthetic_colors[(idx - len(real_sources)) % len(synthetic_colors)]

        y_values = projection_mean_by_source[idx]
        y_errors = projection_std_by_source[idx]
        valid_mask = ~np.isnan(y_values)
        if not np.any(valid_mask):
            continue

        ax_proj.errorbar(
            class_ids[valid_mask],
            y_values[valid_mask],
            yerr=y_errors[valid_mask],
            marker="^",
            linestyle="--",
            linewidth=1.6,
            markersize=4,
            capsize=3,
            color=color,
            alpha=0.9,
            label=f"{source_name}: avg |x · w_c_hat|",
        )

    ax_norm.set_ylabel("||w_c||")
    ax_proj.set_ylabel("avg |x · w_c_hat|")
    ax_norm.set_xlabel("Class")
    ax_norm.set_xticks(class_ids)

    h1, l1 = ax_norm.get_legend_handles_labels()
    h2, l2 = ax_proj.get_legend_handles_labels()
    ax_norm.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)

    fig_align.savefig(str(out_cos_plot), dpi=220, bbox_inches="tight")
    np.savez(
        out_cos_npz,
        class_ids=class_ids,
        source_names=np.array(metric_source_names, dtype=object),
        cosine_by_source=cosine_by_source,
        cosine_std_by_source=cosine_std_by_source,
        class_weight_norms=class_weight_norms,
        projection_mean_by_source=projection_mean_by_source,
        projection_std_by_source=projection_std_by_source,
    )
    log(DEBUG, f"Saved class alignment plot to {out_cos_plot}")
    log(DEBUG, f"Saved class alignment arrays to {out_cos_npz}")

    acc_rows = []
    loss_rows = []
    for source_id, source_name in enumerate(source_names):
        source_labels = source_label_tensors[source_id]
        source_logits = source_logits_tensors[source_id]

        if source_logits.size(0) != source_labels.size(0):
            raise ValueError(
                f"Logit/label size mismatch for {source_name}: "
                f"{source_logits.size(0)} vs {source_labels.size(0)}"
            )

        per_sample_loss = F.cross_entropy(source_logits, source_labels, reduction="none")
        pred_labels = torch.argmax(source_logits, dim=1)

        acc_row = np.full(n_classes, np.nan, dtype=np.float32)
        loss_row = np.full(n_classes, np.nan, dtype=np.float32)
        for class_id in range(n_classes):
            class_mask = source_labels == class_id
            if torch.any(class_mask):
                class_acc = (pred_labels[class_mask] == source_labels[class_mask]).float().mean().item()
                class_loss = per_sample_loss[class_mask].mean().item()
                acc_row[class_id] = class_acc
                loss_row[class_id] = class_loss

        acc_rows.append(acc_row)
        loss_rows.append(loss_row)

    class_accuracy_by_source = np.vstack(acc_rows) if acc_rows else np.empty((0, n_classes), dtype=np.float32)
    class_loss_by_source = np.vstack(loss_rows) if loss_rows else np.empty((0, n_classes), dtype=np.float32)

    fig_cls, axes_cls = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    ax_acc, ax_loss = axes_cls

    for class_id in class_ids:
        ax_acc.axvline(class_id, color="#d9d9d9", linewidth=0.7, zorder=0)
        ax_loss.axvline(class_id, color="#e6e6e6", linewidth=0.7, zorder=0)

    for idx, source_name in enumerate(source_names):
        if idx < len(real_sources):
            color = real_colors[idx % len(real_colors)]
        else:
            color = synthetic_colors[(idx - len(real_sources)) % len(synthetic_colors)]

        acc_values = class_accuracy_by_source[idx]
        acc_mask = ~np.isnan(acc_values)
        if np.any(acc_mask):
            ax_acc.plot(
                class_ids[acc_mask],
                acc_values[acc_mask],
                marker="o",
                linewidth=1.8,
                markersize=4,
                color=color,
                label=source_name,
            )

        loss_values = class_loss_by_source[idx]
        loss_mask = ~np.isnan(loss_values)
        if np.any(loss_mask):
            ax_loss.plot(
                class_ids[loss_mask],
                loss_values[loss_mask],
                marker="s",
                linewidth=1.8,
                markersize=4,
                color=color,
                label=source_name,
            )

    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(-0.02, 1.02)
    ax_acc.set_title("Per-class discriminator accuracy")
    ax_acc.legend(title="Source", loc="best")

    ax_loss.set_ylabel("Cross-Entropy")
    ax_loss.set_xlabel("Class")
    ax_loss.set_title("Per-class discriminator cross-entropy")
    ax_loss.set_xticks(class_ids)

    fig_cls.savefig(str(out_cls_plot), dpi=220, bbox_inches="tight")
    np.savez(
        out_cls_npz,
        class_ids=class_ids,
        source_names=np.array(source_names, dtype=object),
        class_accuracy_by_source=class_accuracy_by_source,
        class_loss_by_source=class_loss_by_source,
    )
    log(DEBUG, f"Saved class accuracy/loss plot to {out_cls_plot}")
    log(DEBUG, f"Saved class accuracy/loss arrays to {out_cls_npz}")

    class_mean_per_source = []
    for source_id, source_name in enumerate(source_names):
        source_labels = source_label_tensors[source_id]
        classifier_repr = source_classifier_repr_tensors[source_id]

        class_means = []
        for class_id in range(n_classes):
            class_mask = source_labels == class_id
            if torch.any(class_mask):
                class_means.append(classifier_repr[class_mask].mean(dim=0))
            else:
                class_means.append(None)
        class_mean_per_source.append(class_means)

    dataset_pair_names = []
    dataset_pair_distance_rows = []
    for left_source_id in range(len(source_names)):
        for right_source_id in range(left_source_id + 1, len(source_names)):
            dataset_pair_names.append(f"{source_names[left_source_id]} vs {source_names[right_source_id]}")
            pair_row = np.full(n_classes, np.nan, dtype=np.float32)
            for class_id in range(n_classes):
                left_vec = class_mean_per_source[left_source_id][class_id]
                right_vec = class_mean_per_source[right_source_id][class_id]
                if left_vec is None or right_vec is None:
                    continue
                pair_row[class_id] = torch.linalg.norm(left_vec - right_vec).item()
            dataset_pair_distance_rows.append(pair_row)

    dataset_pair_distance_by_class = (
        np.vstack(dataset_pair_distance_rows) if dataset_pair_distance_rows else np.empty((0, n_classes), dtype=np.float32)
    )

    fig_dist, ax_dist = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for class_id in class_ids:
        ax_dist.axvline(class_id, color="#e6e6e6", linewidth=0.7, zorder=0)

    for pair_name, pair_row in zip(dataset_pair_names, dataset_pair_distance_by_class):
        valid_mask = ~np.isnan(pair_row)
        if not np.any(valid_mask):
            continue
        ax_dist.plot(
            class_ids[valid_mask],
            pair_row[valid_mask],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=pair_name,
        )

    ax_dist.set_xlabel("Class")
    ax_dist.set_ylabel("Distance between dataset class means")
    ax_dist.set_title("Per-class inter-dataset separation in discriminator feature space")
    ax_dist.set_xticks(class_ids)
    ax_dist.legend(loc="best", fontsize=8)

    fig_dist.savefig(str(out_dist_plot), dpi=220, bbox_inches="tight")
    np.savez(
        out_dist_npz,
        source_names=np.array(source_names, dtype=object),
        dataset_pair_names=np.array(dataset_pair_names, dtype=object),
        dataset_pair_distance_by_class=dataset_pair_distance_by_class,
    )
    log(DEBUG, f"Saved class mean distance plot to {out_dist_plot}")
    log(DEBUG, f"Saved class mean distance arrays to {out_dist_npz}")

    if go is not None:
        plotly_fig = go.Figure()
        for source_id, source_name in enumerate(source_names):
            mask = source_ids == source_id
            if source_id < len(real_sources):
                marker_color = labels[mask]
                marker_dict = dict(
                    size=3,
                    opacity=0.35,
                    color=marker_color,
                    colorscale="Turbo",
                    cmin=0,
                    cmax=9,
                    showscale=(source_id == 0),
                    colorbar=dict(title="Class"),
                )
            else:
                synthetic_index = source_id - len(real_sources)
                marker_dict = dict(
                    size=5,
                    opacity=0.8,
                    color=synthetic_colors[synthetic_index % len(synthetic_colors)],
                )

            plotly_fig.add_trace(
                go.Scatter3d(
                    x=embedding[mask, 0],
                    y=embedding[mask, 1],
                    z=embedding[mask, 2],
                    mode="markers",
                    name=source_name,
                    marker=marker_dict,
                )
            )

        plotly_fig.update_layout(
            title="PCA 3D (fit on real: merged splits; projected synthetic)",
            legend=dict(title="Source"),
            scene=dict(
                xaxis_title="PC1",
                yaxis_title="PC2",
                zaxis_title="PC3",
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        )
        plotly_fig.write_html(str(out_html), include_plotlyjs="cdn")
        log(DEBUG, f"Saved interactive PCA HTML to {out_html}")
    else:
        log(DEBUG, "Plotly not installed; skipping interactive HTML export")

    log(DEBUG, f"Saved PCA plot to {out_plot}")
    log(DEBUG, f"Saved PCA arrays to {out_npz}")


    



def main():
    parser = argparse.ArgumentParser(description="Run experiment for given configuration file.")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of allocated GPUs (default: 1)",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Configuration file path (no default)",
    )
    parser.add_argument(
        "--run-results-path",
        type=str,
        default=None,
        help="Folder containing generator weight files (.pt). Defaults to OUTPUT_CONFIGS.RESULT_LOG_PATH from config.",
    )
    parser.add_argument(
        "--max-samples-per-source",
        type=int,
        default=3000,
        help="Maximum samples to use from each source dataset (split0 and each generated set). Use <=0 for no cap.",
    )
    parser.add_argument(
        "--executor-type",
        type=str,
        default="ProcessPool",       # ThreadPool, ProcessPool
        help="Run clients on thread or process pool (default: ThreadPool)",
    )
    parser.add_argument(
        "--random-reinit",
        action="store_true",
        help="Randomly reinitialize model weights and ignore MODEL_CONFIGS.WEIGHT_PATH",
    )
    args = parser.parse_args()

    user_configs = parse_configs(args.config_file)
    exp_name = ntpath.basename(args.config_file)[:-5]

    # Setup random seeds before anything else
    setup_random_seeds(seed_value=user_configs["SERVER_CONFIGS"]["RANDOM_SEED"])

    # Create stdout re-direction files
    os.makedirs(user_configs["OUTPUT_CONFIGS"]["RESULT_LOG_PATH"], exist_ok=True)
    logfile = open( join(user_configs["OUTPUT_CONFIGS"]["RESULT_LOG_PATH"], f"console_{exp_name}.log"), "w")
    fedml.common.logger.update_console_handler(level=DEBUG, stream=logfile)

    # Setup default torch device
    default_device = "cpu"
    if torch.cuda.is_available(): # and args.executor_type == "ThreadPool":
        default_device = f"cuda:{torch.cuda.current_device()}"
    torch.set_default_device(default_device)

    log(DEBUG, f"# of GPUs       : {args.num_gpus}")
    log(DEBUG, f"Config File     : {args.config_file}")
    log(DEBUG, f"Executor Type   : {args.executor_type}")
    log(DEBUG, f"Random Reinit   : {args.random_reinit}")
    log(DEBUG, f"Run Results Dir : {args.run_results_path if args.run_results_path is not None else user_configs['OUTPUT_CONFIGS']['RESULT_LOG_PATH']}")
    log(DEBUG, f"Max Samples/Src : {args.max_samples_per_source}")

    tSNE(
        exp_name=exp_name,
        user_configs=user_configs,
        num_gpus=args.num_gpus,
        executor_type=args.executor_type,
        run_results_path=args.run_results_path,
        max_samples_per_source=args.max_samples_per_source,
    )

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
