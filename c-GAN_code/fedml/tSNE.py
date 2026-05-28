"""Module to run a single experiment on slurm like environment."""

import warnings
warnings.filterwarnings("ignore")

import multiprocessing
from logging import DEBUG, INFO
from typing import cast, Optional, Any, Tuple

import copy
import torch
import os
from os.path import join
import argparse
import ntpath
import re

import fedml
from fedml.common import log

from fedml.client import create_client
from fedml.configs import parse_configs
from fedml.data_handler import load_and_fetch_split, merge_splits
from fedml.models import load_model
from fedml.modules import ExperimentManager, setup_random_seeds, evaluate_gan, evaluate
from fedml.server import (
    create_server,
    get_client_manager
)
from fedml.strategy import get_strategy
from fedml.defenses.filters.gan_filter import generate_dataset

import numpy as np
from sklearn.decomposition import PCA
import torch.nn.functional as F
import torch.nn as nn
try:
    import plotly.graph_objects as go
except ImportError:
    go = None
try:
    from PIL import Image
except ImportError:
    Image = None

from pathlib import Path




def tSNE(
        exp_name,
        user_configs,
        executor_type,
        num_gpus=None,
        run_results_path: Optional[str] = None,
        max_samples_per_source: Optional[int] = None,
        eval_only: bool = False,
        save_class_grids: bool = False,
        grid_samples_per_class: int = 10,
        run_dis_gen_eval: bool = True,
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

    target_rounds = {54, 55}
    round_pattern = re.compile(r"round[-_](\d+)", re.IGNORECASE)

    def round_in_targets(path: Path) -> bool:
        match = round_pattern.search(path.name)
        if not match:
            return False
        return int(match.group(1)) in target_rounds

    # Keep only generator weight files saved at the end of training.
    gen_model_paths = [
        p for p in file_paths
        if p.suffix == ".pt" and ("gen" in p.name and "_" in p.name) #and round_in_targets(p) # or "gen-attack-weights-" in p.name
    ]

    discriminator_model_paths = [
        p for p in file_paths   if p.suffix == ".pt" and ("global" in p.name and "round" in p.name and "RESNET-18" in p.name) or "SAVE" in p.name #  "weights-global-pre-attack-round-24" , "TEST_DCGAN"
    ]

    def infer_gen_model_name(file_name: str):
        file_name = file_name.upper()
        if "GEN-DCGAN" in file_name:
            return "GEN-DCGAN"
        if "SIGMOID" in file_name:
            return "TEST-SIGMOID"
        if "TANH" in file_name:
            return "TEST-TANH"
        return None
    
    def infer_dis_model_name(file_name: str):
        file_name = file_name.upper()
        if "RESNET-18" in file_name:
            return "RESNET-18-CUSTOM"
        if "RESNET-34" in file_name:
            return "RESNET-34-CUSTOM"
        return None


    mod_conf= {"MODEL_NAME": "TEST-TANH", "NUM_CLASSES": 10, "LATENT_SIZE": 100, "OUT_CHANNEL": 3, "OUTPUT_SIZE": 32, "SAMPLES_PER_CLASS": 1000}
    MODls_configs=[]
    gen_models= []
    gen_source_names = []
    for path in gen_model_paths:
        mod_name = infer_gen_model_name(path.name)
        if mod_name is None:
            continue

        gen_source_name = path.stem

        gen_model_conf = {**mod_conf, "MODEL_NAME": mod_name, "WEIGHT_PATH": str(path)}
        MODls_configs.append(gen_model_conf)

        gen_model = load_model(model_configs=gen_model_conf)
        saved_obj = torch.load(path, map_location="cuda", weights_only=False)
        # if isinstance(saved_obj, dict):
        #     gen_model.load_state_dict(saved_obj)
        # else:
        gen_model.set_weights(saved_obj)
        gen_model.eval()
        gen_models.append(gen_model)
        gen_source_names.append(gen_source_name)

    
    dis_models = []
    dis_model_conf = {**mod_conf, "MODEL_NAME": "RESNET-18-CUSTOM", "WEIGHT_PATH": None}
    for path in discriminator_model_paths:
        mod_name = infer_dis_model_name(path.name)
        if mod_name is None:
            continue

        dis_model_conf = {**mod_conf, "MODEL_NAME": mod_name, "WEIGHT_PATH": str(path)}
        MODls_configs.append(dis_model_conf)

        dis_model = load_model(model_configs=dis_model_conf)
        saved_obj = torch.load(path, map_location="cuda", weights_only=False)
        if isinstance(saved_obj, dict):
            dis_model.load_state_dict(saved_obj)
        else:
            dis_model.set_weights(saved_obj)
        dis_model.eval()
        # store as (name, model) so we can evaluate cross-product later
        dis_models.append((mod_name, dis_model))


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


    num_classes=mod_conf["NUM_CLASSES"]
    samples_per_class=mod_conf["SAMPLES_PER_CLASS"]
    latent_size=mod_conf["LATENT_SIZE"]
    
    # Generate synthetic classes

    # input_znoises = torch.randn(input_classes.size(0), mod_conf["LATENT_SIZE"])
    input_znoises = torch.randn(num_classes * samples_per_class, latent_size)
    input_classes = torch.arange(num_classes, dtype=torch.int64).repeat_interleave(samples_per_class)

    generated_datasets = []
    for model_conf, gen_model, gen_source_name in zip(MODls_configs, gen_models, gen_source_names):
        gen_dataset = generate_dataset(
            gen_model=gen_model,
            input_znoises=input_znoises,
            input_classes=input_classes,
            device="cpu",
            batch_size=1024,
        )
        generated_datasets.append((gen_source_name, gen_dataset))
        log(DEBUG, f"Generated {len(gen_dataset)} samples using {gen_source_name}")

    loader_device = "cuda" if torch.cuda.is_available() else "cpu"
    loader_generator = torch.Generator(device=loader_device)
    generated_dataloaders = [
        (name, torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=True, generator=loader_generator))
        for name, ds in generated_datasets
    ]
    # Prepare output directory for results
    out_dir = Path(user_configs["OUTPUT_CONFIGS"]["RESULT_LOG_PATH"])

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

    def _to_nchw_float01_for_vis(x: torch.Tensor) -> Tuple[torch.Tensor, str]:
        x = x.detach().cpu()
        if x.dim() == 4 and x.shape[-1] in (1, 3) and x.shape[1] not in (1, 3):
            x = x.permute(0, 3, 1, 2)
        x = x.float()
        if x.numel() == 0:
            return x, "empty"

        if x.dim() == 4:
            reduce_dims = (0, 2, 3)
            mean = x.mean(dim=reduce_dims, keepdim=True)
            var = x.var(dim=reduce_dims, unbiased=False, keepdim=True)
        else:
            mean = x.mean()
            var = x.var(unbiased=False)

        std = torch.sqrt(var + 1e-6)
        x = (x - mean) / std

        if x.dim() == 4:
            min_val = x.amin(dim=reduce_dims, keepdim=True)
            max_val = x.amax(dim=reduce_dims, keepdim=True)
        else:
            min_val = x.min()
            max_val = x.max()

        denom = (max_val - min_val).clamp(min=1e-6)
        x = (x - min_val) / denom

        return x.clamp(0.0, 1.0), "batch_norm_minmax"

    def _safe_tag(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())

    def _save_grid_image(grid: np.ndarray, out_path: Path) -> bool:
        if Image is not None:
            Image.fromarray(grid).save(out_path)
            return True
        try:
            import matplotlib.pyplot as plt

            plt.imsave(str(out_path), grid)
            return True
        except Exception:
            return False

    def save_class_grid_image(dataset, source_name: str) -> None:
        if grid_samples_per_class is None or grid_samples_per_class <= 0:
            log(DEBUG, "Grid samples per class <= 0; skipping class grid export")
            return
        data_x, data_y = extract_xy(dataset)
        if data_x.numel() == 0 or data_y.numel() == 0:
            log(DEBUG, f"No data available for class grid export: {source_name}")
            return
        data_x, norm_mode = _to_nchw_float01_for_vis(data_x)
        if data_x.dim() != 4:
            log(DEBUG, f"Expected 4D image tensor for grid export: {source_name}")
            return
        if data_y.dim() != 1:
            data_y = data_y.view(-1)

        if data_x.size(1) == 1:
            data_x = data_x.repeat(1, 3, 1, 1)
        if data_x.size(1) != 3:
            log(DEBUG, f"Unsupported channel count for grid export: {source_name}")
            return

        height = int(data_x.size(2))
        width = int(data_x.size(3))
        grid_height = num_classes * height
        grid_width = grid_samples_per_class * width
        grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        sample_min = float("inf")
        sample_max = float("-inf")
        sample_sum = 0.0
        sample_count = 0

        for class_id in range(num_classes):
            class_idx = torch.nonzero(data_y == class_id, as_tuple=False).view(-1)
            if class_idx.numel() == 0:
                continue
            if class_idx.numel() > grid_samples_per_class:
                pick = sampling_rng.choice(class_idx.numel(), size=grid_samples_per_class, replace=False)
                pick_idx = torch.as_tensor(pick, dtype=torch.long, device=class_idx.device)
                class_idx = class_idx[pick_idx]
            else:
                class_idx = class_idx[:grid_samples_per_class]
            class_imgs = data_x[class_idx].clamp(0.0, 1.0)
            sample_min = min(sample_min, class_imgs.min().item())
            sample_max = max(sample_max, class_imgs.max().item())
            sample_sum += class_imgs.mean().item() * class_imgs.numel()
            sample_count += class_imgs.numel()
            class_imgs_u8 = class_imgs.mul(255.0).to(torch.uint8)
            for col_idx in range(class_imgs_u8.size(0)):
                img_np = class_imgs_u8[col_idx].permute(1, 2, 0).cpu().numpy()
                y0 = class_id * height
                x0 = col_idx * width
                grid[y0:y0 + height, x0:x0 + width] = img_np

        if sample_count > 0:
            sample_mean = sample_sum / sample_count
            log(DEBUG, f"Grid norm {source_name}: mode={norm_mode}")
            log(DEBUG, f"Grid sample stats {source_name}: min={sample_min:.4f}, max={sample_max:.4f}, mean={sample_mean:.4f}")

        out_grid = out_dir / f"class_grid_{_safe_tag(source_name)}_{exp_name}.png"
        if _save_grid_image(grid, out_grid):
            log(DEBUG, f"Saved class grid image to {out_grid}")
        else:
            log(DEBUG, f"Could not save class grid image (missing PIL/matplotlib): {out_grid}")

    def _entropy_from_probs(probs: np.ndarray) -> float:
        probs = probs[probs > 0]
        if probs.size == 0:
            return 0.0
        return float(-(probs * np.log(probs)).sum())

    def compute_diversity_metrics(
        features_np: np.ndarray,
        labels_np: np.ndarray,
        num_classes: int,
        rng: Any,
        max_pairwise_samples: int = 1000,
    ) -> Tuple[dict, np.ndarray]:
        nan = float("nan")
        metrics = {
            "n_samples": 0.0,
            "class_coverage": nan,
            "coverage_ratio": nan,
            "class_entropy": nan,
            "effective_num_classes": nan,
            "feature_var_mean": nan,
            "feature_norm_mean": nan,
            "within_class_var_mean": nan,
            "between_class_mean_l2": nan,
            "pairwise_l2_mean": nan,
            "pairwise_cosine_dist_mean": nan,
        }

        features = np.asarray(features_np)
        labels = np.asarray(labels_np).astype(np.int64, copy=False)
        if features.ndim == 1:
            features = features.reshape(-1, 1)

        n_samples = int(features.shape[0]) if features.size else 0
        metrics["n_samples"] = float(n_samples)
        if n_samples == 0 or num_classes <= 0:
            return metrics, np.zeros((max(num_classes, 0),), dtype=np.int64)

        class_counts = np.bincount(labels, minlength=num_classes)
        if class_counts.size > num_classes:
            class_counts = class_counts[:num_classes]

        total = float(class_counts.sum())
        if total > 0:
            probs = class_counts.astype(np.float64) / total
            entropy = _entropy_from_probs(probs)
            if num_classes > 1:
                metrics["class_entropy"] = entropy / float(np.log(num_classes))
            else:
                metrics["class_entropy"] = 0.0
            metrics["effective_num_classes"] = float(np.exp(entropy))
        else:
            probs = np.zeros((num_classes,), dtype=np.float64)

        class_coverage = int((class_counts > 0).sum())
        metrics["class_coverage"] = float(class_coverage)
        metrics["coverage_ratio"] = float(class_coverage) / float(num_classes) if num_classes > 0 else nan

        if n_samples >= 2:
            metrics["feature_var_mean"] = float(np.var(features, axis=0).mean())
        metrics["feature_norm_mean"] = float(np.linalg.norm(features, axis=1).mean())

        within_vars = []
        for class_id in range(num_classes):
            class_mask = labels == class_id
            if class_mask.sum() >= 2:
                within_vars.append(np.var(features[class_mask], axis=0).mean())
        if within_vars:
            metrics["within_class_var_mean"] = float(np.mean(within_vars))

        class_means = []
        for class_id in range(num_classes):
            class_mask = labels == class_id
            if class_mask.sum() > 0:
                class_means.append(features[class_mask].mean(axis=0))
        if len(class_means) >= 2:
            means_tensor = torch.as_tensor(np.stack(class_means), dtype=torch.float32)
            mean_dist = torch.pdist(means_tensor, p=2)
            if mean_dist.numel() > 0:
                metrics["between_class_mean_l2"] = float(mean_dist.mean().item())

        sample_n = min(max_pairwise_samples, n_samples)
        if sample_n >= 2:
            if sample_n < n_samples:
                sample_idx = rng.choice(n_samples, size=sample_n, replace=False)
                sample_feat = features[sample_idx]
            else:
                sample_feat = features

            sample_tensor = torch.as_tensor(sample_feat, dtype=torch.float32)
            pairwise_l2 = torch.pdist(sample_tensor, p=2)
            if pairwise_l2.numel() > 0:
                metrics["pairwise_l2_mean"] = float(pairwise_l2.mean().item())

            norms = torch.linalg.norm(sample_tensor, dim=1, keepdim=True).clamp(min=1e-12)
            normed = sample_tensor / norms
            cos_matrix = normed @ normed.T
            tri_idx = torch.triu_indices(sample_n, sample_n, offset=1)
            cos_vals = cos_matrix[tri_idx[0], tri_idx[1]]
            if cos_vals.numel() > 0:
                metrics["pairwise_cosine_dist_mean"] = float(1.0 - cos_vals.mean().item())

        return metrics, class_counts

    if max_samples_per_source is not None and max_samples_per_source <= 0:
        max_samples_per_source = None

    sampling_rng = make_rng(seed=user_configs["SERVER_CONFIGS"]["RANDOM_SEED"])

    if save_class_grids:
        save_class_grid_image(real_split, "real")
        for gen_name, gen_dataset in generated_datasets:
            save_class_grid_image(gen_dataset, f"gen_{gen_name}")

    if len(generated_datasets) == 0:
        log(DEBUG, "No generator datasets found to run t-SNE")
        return

    if len(dis_models) == 0:
        log(DEBUG, "No discriminator models found; skipping evaluations")
        return

    if run_dis_gen_eval:
        # Evaluate Cartesian product: every discriminator vs every generated dataset
        n_dis = len(dis_models)
        n_gen = len(generated_dataloaders)
        dis_names = [name for name, _ in dis_models]
        gen_names = [name for name, _ in generated_dataloaders]
        acc_matrix = np.full((n_dis, n_gen), np.nan, dtype=np.float32)
        loss_matrix = np.full((n_dis, n_gen), np.nan, dtype=np.float32)

        for i, (dname, dmodel) in enumerate(dis_models):
            for j, (gname, gloader) in enumerate(generated_dataloaders):
                stats = evaluate(
                    model=dmodel,
                    testloader=gloader,
                    device="cuda",
                    criterion=nn.CrossEntropyLoss(),
                )
                loss_matrix[i, j] = stats[0]
                acc_matrix[i, j] = stats[1]
                log(INFO, f"Eval dis={dname} on gen={gname}: loss={stats[0]:.4f}, accuracy={stats[1]:.4f}")

        out_eval_npz = out_dir / f"dis_vs_gen_eval_{exp_name}.npz"
        np.savez(
            out_eval_npz,
            dis_names=np.array(dis_names, dtype=object),
            gen_names=np.array(gen_names, dtype=object),
            loss=loss_matrix,
            accuracy=acc_matrix,
        )
        log(DEBUG, f"Saved discriminator vs generator evaluation matrix to {out_eval_npz}")

        # Save CSV summaries and heatmap visualizations for quick inspection.
        try:
            import csv

            out_eval_csv = out_dir / f"dis_vs_gen_eval_{exp_name}.csv"
            with open(out_eval_csv, "w", newline="") as csvf:
                writer = csv.writer(csvf)
                writer.writerow(["discriminator\\generator"] + gen_names)
                for i, dname in enumerate(dis_names):
                    row = [dname] + [f"{acc_matrix[i, j]:.4f}" if not np.isnan(acc_matrix[i, j]) else "" for j in range(n_gen)]
                    writer.writerow(row)
            log(DEBUG, f"Saved discriminator vs generator accuracy CSV to {out_eval_csv}")

            out_loss_csv = out_dir / f"dis_vs_gen_loss_{exp_name}.csv"
            with open(out_loss_csv, "w", newline="") as csvf:
                writer = csv.writer(csvf)
                writer.writerow(["discriminator\\generator"] + gen_names)
                for i, dname in enumerate(dis_names):
                    row = [dname] + [f"{loss_matrix[i, j]:.4f}" if not np.isnan(loss_matrix[i, j]) else "" for j in range(n_gen)]
                    writer.writerow(row)
            log(DEBUG, f"Saved discriminator vs generator loss CSV to {out_loss_csv}")
        except Exception:
            log(DEBUG, "Could not save CSV summaries for eval matrices")

        if go is not None:
            try:
                # Interactive heatmap for accuracy using Plotly
                fig_acc = go.Figure(
                    data=go.Heatmap(z=acc_matrix.tolist(), x=list(gen_names), y=list(dis_names), colorscale="Viridis", zmin=0.0, zmax=1.0)
                )
                fig_acc.update_layout(title="Discriminator vs Generator Accuracy", xaxis=dict(tickangle=-45))
                out_eval_acc_html = out_dir / f"dis_vs_gen_accuracy_{exp_name}.html"
                fig_acc.write_html(str(out_eval_acc_html), include_plotlyjs="cdn")
                log(DEBUG, f"Saved discriminator vs generator accuracy interactive HTML to {out_eval_acc_html}")

                # Interactive heatmap for loss (None for NaN)
                loss_z = np.where(np.isfinite(loss_matrix), loss_matrix, None).tolist()
                fig_loss = go.Figure(
                    data=go.Heatmap(z=loss_z, x=list(gen_names), y=list(dis_names), colorscale="Magma")
                )
                fig_loss.update_layout(title="Discriminator vs Generator Loss", xaxis=dict(tickangle=-45))
                out_eval_loss_html = out_dir / f"dis_vs_gen_loss_{exp_name}.html"
                fig_loss.write_html(str(out_eval_loss_html), include_plotlyjs="cdn")
                log(DEBUG, f"Saved discriminator vs generator loss interactive HTML to {out_eval_loss_html}")
            except Exception:
                log(DEBUG, "Could not save eval heatmap visualizations")
        else:
            log(INFO, "Plotly is not installed; skipping interactive eval heatmaps")
            # continue but skip plotting sections below
    else:
        log(INFO, "Skipping discriminator vs generator evaluations (disabled)")

    # If caller only requested evaluation, stop here
    if eval_only:
        if run_dis_gen_eval:
            log(INFO, "Eval-only requested; skipping feature extraction and plotting")
        else:
            log(INFO, "Eval-only requested but evaluations disabled; skipping downstream work")
        return

    # keep first discriminator as primary for downstream representation extraction
    primary_dis_name, dis_model = dis_models[0]

    # Build last-layer feature representations over real split0 plus generated datasets.
    

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

    diversity_rows = []
    diversity_class_counts = []
    for source_id, source_name in enumerate(source_names):
        metrics, class_counts = compute_diversity_metrics(
            features_np=all_x[source_id],
            labels_np=all_y[source_id],
            num_classes=num_classes,
            rng=sampling_rng,
        )
        metrics["source_name"] = source_name
        metrics["source_type"] = "real" if source_id < len(real_sources) else "gen"
        diversity_rows.append(metrics)
        diversity_class_counts.append(class_counts)

    if diversity_rows:
        try:
            import csv

            metric_fields = [
                "n_samples",
                "class_coverage",
                "coverage_ratio",
                "class_entropy",
                "effective_num_classes",
                "feature_var_mean",
                "feature_norm_mean",
                "within_class_var_mean",
                "between_class_mean_l2",
                "pairwise_l2_mean",
                "pairwise_cosine_dist_mean",
            ]

            out_div_csv = out_dir / f"diversity_metrics_{exp_name}.csv"
            with open(out_div_csv, "w", newline="") as csvf:
                writer = csv.writer(csvf)
                writer.writerow(["source", "source_type"] + metric_fields)
                for row in diversity_rows:
                    writer.writerow(
                        [row["source_name"], row["source_type"]]
                        + ["" if np.isnan(row[field]) else f"{row[field]:.6f}" for field in metric_fields]
                    )
            log(DEBUG, f"Saved diversity metrics CSV to {out_div_csv}")

            out_div_npz = out_dir / f"diversity_metrics_{exp_name}.npz"
            metrics_matrix = np.array(
                [[row[field] for field in metric_fields] for row in diversity_rows],
                dtype=np.float32,
            )
            np.savez(
                out_div_npz,
                source_names=np.array([row["source_name"] for row in diversity_rows], dtype=object),
                source_types=np.array([row["source_type"] for row in diversity_rows], dtype=object),
                metric_names=np.array(metric_fields, dtype=object),
                metrics=metrics_matrix,
                class_counts=np.stack(diversity_class_counts) if diversity_class_counts else np.empty((0, num_classes), dtype=np.int64),
            )
            log(DEBUG, f"Saved diversity metrics NPZ to {out_div_npz}")
        except Exception:
            log(DEBUG, "Could not save diversity metrics")

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

    out_plot = out_dir / f"pca3_lastlayer_realfit_projected_{exp_name}.html"
    out_npz = out_dir / f"pca3_lastlayer_realfit_projected_{exp_name}.npz"
    out_html = out_dir / f"pca3_lastlayer_realfit_projected_{exp_name}.html"
    out_cos_npz = out_dir / f"class_alignment_cosine_norm_{exp_name}.npz"
    out_cls_npz = out_dir / f"class_accuracy_crossentropy_{exp_name}.npz"
    out_dist_npz = out_dir / f"class_mean_distance_{exp_name}.npz"
    # Save arrays for plotting/inspection; interactive plots use Plotly below when available
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

    # Save class alignment arrays (plots generated with Plotly below when available)
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

    # Save class accuracy/loss arrays (interactive plots with Plotly when available)
    np.savez(
        out_cls_npz,
        class_ids=class_ids,
        source_names=np.array(source_names, dtype=object),
        class_accuracy_by_source=class_accuracy_by_source,
        class_loss_by_source=class_loss_by_source,
    )
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

    # Save class mean distance arrays (interactive plots with Plotly when available)
    np.savez(
        out_dist_npz,
        source_names=np.array(source_names, dtype=object),
        dataset_pair_names=np.array(dataset_pair_names, dtype=object),
        dataset_pair_distance_by_class=dataset_pair_distance_by_class,
    )
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
        # Also produce additional interactive plots (class alignment, accuracy/loss, mean distances)
        try:
            from plotly.subplots import make_subplots

            # Class alignment (cosine + projections)
            fig_align = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=(
                "Class-wise alignment: class weight vs class mean embedding",
                "Class weight norms and avg |x · w_c_hat| per class",
            ))
            for idx, source_name in enumerate(metric_source_names):
                color = real_colors[idx % len(real_colors)] if idx < len(real_sources) else synthetic_colors[(idx - len(real_sources)) % len(synthetic_colors)]
                y = cosine_by_source[idx] if cosine_by_source.size else np.array([])
                mask = ~np.isnan(y)
                if mask.any():
                    fig_align.add_trace(
                        go.Scatter(x=class_ids[mask].tolist(), y=y[mask].tolist(), mode="lines+markers", name=source_name, line=dict(color=color)),
                        row=1, col=1
                    )

                pm = projection_mean_by_source[idx] if projection_mean_by_source.size else np.array([])
                ps = projection_std_by_source[idx] if projection_std_by_source.size else np.array([])
                mask2 = ~np.isnan(pm)
                if mask2.any():
                    fig_align.add_trace(
                        go.Scatter(x=class_ids[mask2].tolist(), y=pm[mask2].tolist(), mode="lines+markers", name=f"{source_name}: avg |x·w|", line=dict(color=color)),
                        row=2, col=1
                    )

            out_align_html = out_dir / f"class_alignment_{exp_name}.html"
            fig_align.update_layout(height=700)
            fig_align.write_html(str(out_align_html), include_plotlyjs="cdn")
            log(DEBUG, f"Saved interactive class alignment HTML to {out_align_html}")
        except Exception:
            log(DEBUG, "Could not create Plotly class alignment plot")

        try:
            # Per-class accuracy and loss
            fig_cls = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Per-class discriminator accuracy", "Per-class discriminator cross-entropy"))
            for idx, source_name in enumerate(source_names):
                color = real_colors[idx % len(real_colors)] if idx < len(real_sources) else synthetic_colors[(idx - len(real_sources)) % len(synthetic_colors)]
                acc_vals = class_accuracy_by_source[idx] if class_accuracy_by_source.size else np.array([])
                loss_vals = class_loss_by_source[idx] if class_loss_by_source.size else np.array([])
                maska = ~np.isnan(acc_vals)
                maskl = ~np.isnan(loss_vals)
                if maska.any():
                    fig_cls.add_trace(go.Scatter(x=class_ids[maska].tolist(), y=acc_vals[maska].tolist(), mode="lines+markers", name=source_name, line=dict(color=color)), row=1, col=1)
                if maskl.any():
                    fig_cls.add_trace(go.Scatter(x=class_ids[maskl].tolist(), y=loss_vals[maskl].tolist(), mode="lines+markers", name=source_name, line=dict(color=color)), row=2, col=1)
            out_cls_html = out_dir / f"class_accuracy_crossentropy_{exp_name}.html"
            fig_cls.update_layout(height=700)
            fig_cls.write_html(str(out_cls_html), include_plotlyjs="cdn")
            log(DEBUG, f"Saved interactive class accuracy/loss HTML to {out_cls_html}")
        except Exception:
            log(DEBUG, "Could not create Plotly class accuracy/loss plot")

        try:
            # Per-class inter-dataset separation
            fig_dist = go.Figure()
            for pair_name, pair_row in zip(dataset_pair_names, dataset_pair_distance_by_class):
                mask = ~np.isnan(pair_row)
                if not mask.any():
                    continue
                fig_dist.add_trace(go.Scatter(x=class_ids[mask].tolist(), y=pair_row[mask].tolist(), mode="lines+markers", name=pair_name))
            out_dist_html = out_dir / f"class_mean_distance_{exp_name}.html"
            fig_dist.update_layout(title="Per-class inter-dataset separation in discriminator feature space", height=400)
            fig_dist.write_html(str(out_dist_html), include_plotlyjs="cdn")
            log(DEBUG, f"Saved interactive class mean distance HTML to {out_dist_html}")
        except Exception:
            log(DEBUG, "Could not create Plotly class mean distance plot")
    else:
        log(DEBUG, "Plotly not installed; skipping interactive HTML export")

    log(DEBUG, f"Saved PCA plot to {out_plot}")
    log(DEBUG, f"Saved PCA arrays to {out_npz}")

    # Compute dominant directions (PCA) for each layer and plot variance explained
    try:
        # Identify available layers in discriminator
        available_layers = []
        if hasattr(dis_model, "features"):
            # For ResNet-like models with features() method
            available_layers = list(range(1, 6))  # layers 1-5
        
        if not available_layers:
            log(DEBUG, "Cannot extract per-layer features; skipping variance analysis")
        else:
            n_layers = len(available_layers)
            n_sources = len(source_names)
            
            # Extract representations from each layer for each dataset
            layer_reprs_by_source = []  # [source][layer] -> tensor of shape (n_samples, n_features)
            
            for source_id, dataset in enumerate(all_datasets):
                data_x, _ = extract_xy(dataset)
                data_x = _to_nchw_float01(data_x)
                
                layer_reprs = []
                for layer_id in available_layers:
                    layer_repr = extract_last_layer_repr(
                        model=dis_model,
                        data_x=data_x,
                        batch_size=1024,
                        layer=layer_id
                    )
                    layer_reprs.append(layer_repr.numpy())
                layer_reprs_by_source.append(layer_reprs)
            
            layer_variance_data = {}
            
            # Use Plotly if available, else fall back to matplotlib
            if go is not None:
                from plotly.subplots import make_subplots
                
                n_rows = (n_layers + 1) // 2
                n_cols = 2
                fig_plotly = make_subplots(
                    rows=n_rows,
                    cols=n_cols,
                    subplot_titles=[f"Layer {lid}" for lid in available_layers],
                    specs=[[{"secondary_y": False} for _ in range(n_cols)] for _ in range(n_rows)],
                )
                
                for layer_idx, layer_id in enumerate(available_layers):
                    row = (layer_idx // n_cols) + 1
                    col = (layer_idx % n_cols) + 1
                    
                    layer_variance_data[layer_id] = {}
                    
                    for source_id, source_name in enumerate(source_names):
                        layer_repr = layer_reprs_by_source[source_id][layer_idx]
                        n_components = min(layer_repr.shape[0], layer_repr.shape[1], 50)
                        
                        pca_layer = PCA(n_components=n_components)
                        pca_layer.fit(layer_repr)
                        
                        cumsum_var = np.cumsum(pca_layer.explained_variance_ratio_)
                        k_values = np.arange(1, len(cumsum_var) + 1)
                        
                        if source_id < len(real_sources):
                            color = real_colors[source_id % len(real_colors)]
                        else:
                            color = synthetic_colors[(source_id - len(real_sources)) % len(synthetic_colors)]
                        
                        fig_plotly.add_trace(
                            go.Scatter(
                                x=k_values,
                                y=cumsum_var,
                                mode="lines+markers",
                                name=source_name,
                                line=dict(color=color, width=2),
                                marker=dict(size=4),
                                showlegend=(layer_idx == 0),  # Only show legend for first layer
                            ),
                            row=row,
                            col=col,
                        )
                        
                        layer_variance_data[layer_id][source_name] = {
                            "cumsum_variance": cumsum_var,
                            "k_values": k_values,
                            "explained_variance_ratio": pca_layer.explained_variance_ratio_,
                        }
                    
                    # Set axis labels
                    fig_plotly.update_xaxes(title_text="Number of Components (k)", row=row, col=col)
                    fig_plotly.update_yaxes(title_text="Cumulative Variance Explained", row=row, col=col, range=[0, 1.05])
                
                fig_plotly.update_layout(
                    title_text="Layer-wise Variance Explained vs Number of Principal Components",
                    height=300 * n_rows,
                    showlegend=True,
                    legend=dict(x=1.02, y=1),
                    hovermode="x unified",
                )
                
                out_var_html = out_dir / f"layer_variance_explained_{exp_name}.html"
                fig_plotly.write_html(str(out_var_html), include_plotlyjs="cdn")
                log(DEBUG, f"Saved interactive layer variance explained HTML to {out_var_html}")
            
            else:
                # Plotly not available; saved per-layer variance arrays already
                log(DEBUG, "Plotly not available for layer variance plotting; saved raw arrays only")
            
            # Save variance data
            out_var_npz = out_dir / f"layer_variance_explained_{exp_name}.npz"
            np.savez(str(out_var_npz), **{
                f"layer_{lid}_{src_name}_{key}": val
                for lid, layer_dict in layer_variance_data.items()
                for src_name, src_dict in layer_dict.items()
                for key, val in src_dict.items()
            })
            log(DEBUG, f"Saved layer variance explained data to {out_var_npz}")
            
    except Exception as e:
        log(DEBUG, f"Could not compute per-layer variance analysis: {e}")
    


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
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run discriminator×generator evaluations and exit (skip feature extraction/plots).",
    )
    parser.add_argument(
        "--save-class-grids",
        action="store_true",
        help="Save per-class image grids for real and generated datasets.",
    )
    parser.add_argument(
        "--grid-samples-per-class",
        type=int,
        default=10,
        help="Number of samples per class in each grid (default: 10).",
    )
    parser.add_argument(
        "--skip-dis-gen-eval",
        action="store_true",
        help="Skip discriminator vs generator accuracy/loss evaluations.",
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
    log(DEBUG, f"Save Grids      : {args.save_class_grids}")
    log(DEBUG, f"Grid Samples    : {args.grid_samples_per_class}")
    log(DEBUG, f"Dis/Gen Eval    : {not args.skip_dis_gen_eval}")

    tSNE(
        exp_name=exp_name,
        user_configs=user_configs,
        num_gpus=args.num_gpus,
        executor_type=args.executor_type,
        run_results_path=args.run_results_path,
        max_samples_per_source=args.max_samples_per_source,
        eval_only=args.eval_only,
        save_class_grids=args.save_class_grids,
        grid_samples_per_class=args.grid_samples_per_class,
        run_dis_gen_eval=not args.skip_dis_gen_eval,
    )

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
