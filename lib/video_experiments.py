"""
Experiment orchestration and hyperparameter search for the FVC video classifier.

Includes:
- Grid search over key hyperparameters (learning rate, weight decay, epochs, freezing)
- Utility to build model and loaders from metadata CSV
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import itertools
import os
import logging

import polars as pl
import torch
from torch.utils.data import DataLoader

from .video_modeling import (
    VideoConfig,
    VideoDataset,
    variable_ar_collate,
    VariableARVideoModel,
)
from .video_training import (
    OptimConfig,
    TrainConfig,
    freeze_backbone_unfreeze_head,
    fit,
)
from .video_metrics import collect_logits_and_labels, basic_classification_metrics
from .video_data import load_metadata, SplitConfig, train_val_test_split, stratified_kfold


logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    project_root: str
    csv_path: str
    batch_size: int = 4
    num_workers: int = 0
    val_split: float = 0.2
    intermediate_dir: str = "data/intermediates"
    # Video config
    num_frames: int = 16
    # Training search space
    lrs: Iterable[float] = (1e-4, 3e-4)
    weight_decays: Iterable[float] = (1e-4, 1e-5)
    num_epochs_list: Iterable[int] = (10, 20)
    freeze_backbone_options: Iterable[bool] = (False, True)


def build_loaders_from_csv(
    cfg: ExperimentConfig,
) -> Dict[str, DataLoader]:
    """Build train/val loaders from the metadata CSV."""
    meta_df = load_metadata(cfg.csv_path)

    logger.info("Loaded metadata with %d rows", meta_df.height)

    # Train/val/test at DataFrame level, saved to Arrow for reproducibility.
    splits = train_val_test_split(
        meta_df,
        SplitConfig(val_size=cfg.val_split, test_size=0.0),
        save_dir=os.path.join(cfg.project_root, cfg.intermediate_dir),
    )
    train_df = splits["train"]
    val_df = splits["val"]

    logger.info("Train size: %d, Val size: %d", train_df.height, val_df.height)

    video_cfg = VideoConfig(num_frames=cfg.num_frames)
    train_ds = VideoDataset(train_df, cfg.project_root, config=video_cfg, train=True)
    val_ds = VideoDataset(val_df, cfg.project_root, config=video_cfg, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=variable_ar_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=variable_ar_collate,
    )

    return {"train": train_loader, "val": val_loader}


def run_grid_search(cfg: ExperimentConfig) -> List[Dict]:
    """
    Run a simple grid search over a few hyperparameters.

    Returns a list of result dicts sorted by best F1 on validation set.
    """
    loaders = build_loaders_from_csv(cfg)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results: List[Dict] = []

    grid = itertools.product(
        cfg.lrs,
        cfg.weight_decays,
        cfg.num_epochs_list,
        cfg.freeze_backbone_options,
    )

    for lr, wd, num_epochs, freeze_backbone_flag in grid:
        logger.info(
            "Grid config: lr=%s, wd=%s, epochs=%s, freeze_backbone=%s",
            lr,
            wd,
            num_epochs,
            freeze_backbone_flag,
        )
        model = VariableARVideoModel()
        if freeze_backbone_flag:
            # Example transfer-learning behavior
            freeze_backbone_unfreeze_head(model)

        optim_cfg = OptimConfig(lr=lr, weight_decay=wd)
        train_cfg = TrainConfig(num_epochs=num_epochs, device=device)

        # Use a subdirectory per grid configuration for checkpoints if desired.
        ckpt_dir = os.path.join(
            cfg.project_root,
            cfg.intermediate_dir,
            "grid_search",
            f"lr{lr}_wd{wd}_ep{num_epochs}_freeze{freeze_backbone_flag}",
        )
        train_cfg = TrainConfig(
            num_epochs=num_epochs,
            device=device,
            checkpoint_dir=ckpt_dir,
        )

        model = fit(
            model,
            train_loader=train_loader,
            val_loader=val_loader,
            optim_cfg=optim_cfg,
            train_cfg=train_cfg,
        )

        logits, labels = collect_logits_and_labels(model, val_loader, device=device)
        metrics = basic_classification_metrics(logits, labels)

        result = {
            "lr": lr,
            "weight_decay": wd,
            "num_epochs": num_epochs,
            "freeze_backbone": freeze_backbone_flag,
            **metrics,
        }
        results.append(result)

    # Sort by F1 descending
    results.sort(key=lambda r: r.get("f1", 0.0), reverse=True)

    # Persist full grid search results to Arrow/Feather by default.
    if cfg.intermediate_dir:
        os.makedirs(os.path.join(cfg.project_root, cfg.intermediate_dir), exist_ok=True)
        results_df = pl.DataFrame(results)
        results_df.write_ipc(
            os.path.join(cfg.project_root, cfg.intermediate_dir, "grid_search_results.feather")
        )

    return results


def run_kfold_experiments(
    cfg: ExperimentConfig,
    n_splits: int = 5,
) -> List[Dict]:
    """
    Run stratified K-fold experiments and save per-fold metrics to Arrow.
    """
    meta_df = load_metadata(cfg.csv_path)
    folds = stratified_kfold(meta_df, n_splits=n_splits)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    video_cfg = VideoConfig(num_frames=cfg.num_frames)

    all_results: List[Dict] = []

    for fold_idx, (train_df, val_df) in enumerate(folds):
        logger.info("Fold %d/%d", fold_idx + 1, n_splits)

        train_ds = VideoDataset(train_df, cfg.project_root, config=video_cfg, train=True)
        val_ds = VideoDataset(val_df, cfg.project_root, config=video_cfg, train=False)

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=variable_ar_collate,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=variable_ar_collate,
        )

        model = VariableARVideoModel()
        optim_cfg = OptimConfig()
        ckpt_dir = os.path.join(
            cfg.project_root,
            cfg.intermediate_dir,
            "kfold",
            f"fold_{fold_idx+1}",
        )
        train_cfg = TrainConfig(
            num_epochs=cfg.num_epochs_list[-1] if hasattr(cfg, "num_epochs_list") else 10,
            device=device,
            checkpoint_dir=ckpt_dir,
        )

        model = fit(
            model,
            train_loader=train_loader,
            val_loader=val_loader,
            optim_cfg=optim_cfg,
            train_cfg=train_cfg,
        )

        logits, labels = collect_logits_and_labels(model, val_loader, device=device)
        metrics = basic_classification_metrics(logits, labels)
        fold_result = {"fold": fold_idx + 1, **metrics}
        all_results.append(fold_result)

    if cfg.intermediate_dir:
        base = os.path.join(cfg.project_root, cfg.intermediate_dir)
        os.makedirs(base, exist_ok=True)
        df = pl.DataFrame(all_results)
        df.write_ipc(os.path.join(base, "kfold_results.feather"))

    return all_results


__all__ = [
    "ExperimentConfig",
    "build_loaders_from_csv",
    "run_grid_search",
    "run_kfold_experiments",
]


