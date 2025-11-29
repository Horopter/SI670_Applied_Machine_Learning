"""
Model comparison and evaluation utilities.

Compare all models on the same test set and generate comparison reports.
"""

from __future__ import annotations

import os
import logging
from typing import Dict, List
import numpy as np
import polars as pl
import pandas as pd
import torch

from .video_modeling import VideoConfig, VideoDataset, variable_ar_collate
from .model_factory import create_model, is_pytorch_model

from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def evaluate_all_models(
    model_types: List[str],
    test_df: pl.DataFrame,
    project_root: str,
    config: Dict[str, any],
    models_dir: str
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all models on the same test set.
    
    Args:
        model_types: List of model types to evaluate
        test_df: Test DataFrame
        project_root: Project root directory
        config: Configuration dictionary (RunConfig-like)
        models_dir: Directory containing trained models
    
    Returns:
        Dictionary mapping model_type to metrics dictionary
    """
    results = {}
    
    video_cfg = VideoConfig(
        num_frames=config.get("num_frames", 8),
        fixed_size=config.get("fixed_size", 224),
        augmentation_config=None,  # No augmentation for evaluation
        temporal_augmentation_config=None,
    )
    
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    
    for model_type in model_types:
        logger.info("Evaluating model: %s", model_type)
        
        try:
            if is_pytorch_model(model_type):
                # PyTorch model evaluation
                model = create_model(model_type, config)
                
                # Load best model from checkpoint
                checkpoint_dir = os.path.join(models_dir, model_type)
                best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
                
                if os.path.exists(best_model_path):
                    checkpoint = torch.load(best_model_path, map_location=device)
                    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
                    logger.info("Loaded checkpoint from %s", best_model_path)
                else:
                    logger.warning("No checkpoint found for %s. Using untrained model.", model_type)
                
                model = model.to(device)
                model.eval()
                
                # Create dataset and loader
                test_ds = VideoDataset(test_df, project_root, config=video_cfg, train=False)
                # Handle both RunConfig and dict
                if hasattr(config, "batch_size"):
                    batch_size = config.batch_size
                    num_workers = config.num_workers
                else:
                    batch_size = config.get("batch_size", 8)
                    num_workers = config.get("num_workers", 2)
                
                test_loader = DataLoader(
                    test_ds,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=torch.cuda.is_available(),
                    collate_fn=variable_ar_collate,
                )
                
                # Evaluate
                from .video_metrics import collect_logits_and_labels, basic_classification_metrics
                
                with torch.no_grad():
                    logits, labels = collect_logits_and_labels(model, test_loader, device=device)
                    metrics = basic_classification_metrics(logits, labels)
                
                results[model_type] = metrics
                
                # Cleanup
                del model, test_loader, test_ds
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            else:
                # Baseline model evaluation
                model = create_model(model_type, config)
                
                # Load model
                model_dir = os.path.join(models_dir, model_type)
                if os.path.exists(model_dir):
                    model.load(model_dir)
                else:
                    logger.warning("No saved model found for %s", model_type)
                    continue
                
                # Predict
                probs = model.predict(test_df, project_root)
                
                # Get labels
                label_map = {label: idx for idx, label in enumerate(sorted(set(test_df["label"].to_list())))}
                labels = np.array([label_map[label] for label in test_df["label"].to_list()])
                preds = np.argmax(probs, axis=1)
                
                # Compute metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                metrics = {
                    "accuracy": float(accuracy_score(labels, preds)),
                    "precision": float(precision_score(labels, preds, average='binary', zero_division=0)),
                    "recall": float(recall_score(labels, preds, average='binary', zero_division=0)),
                    "f1": float(f1_score(labels, preds, average='binary', zero_division=0)),
                }
                
                results[model_type] = metrics
            
            logger.info("%s results: %s", model_type, results[model_type])
            
        except Exception as e:
            logger.error("Failed to evaluate %s: %s", model_type, str(e))
            results[model_type] = {"error": str(e)}
    
    return results


def compare_models(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Create a comparison DataFrame from model results.
    
    Args:
        results: Dictionary mapping model_type to metrics
    
    Returns:
        pandas DataFrame with model comparison
    """
    # Filter out models with errors
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    
    if not valid_results:
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(valid_results).T
    
    # Sort by accuracy (or F1 if available)
    if "accuracy" in df.columns:
        df = df.sort_values("accuracy", ascending=False)
    elif "f1" in df.columns:
        df = df.sort_values("f1", ascending=False)
    
    return df


def plot_model_comparison(results: Dict[str, Dict[str, float]], output_path: str) -> None:
    """
    Plot model comparison (if matplotlib is available).
    
    Args:
        results: Dictionary mapping model_type to metrics
        output_path: Path to save plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available. Skipping plot generation.")
        return
    
    df = compare_models(results)
    
    if df.empty:
        logger.warning("No valid results to plot.")
        return
    
    # Create bar plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metrics_to_plot = ["accuracy", "precision", "recall", "f1"]
    
    for idx, metric in enumerate(metrics_to_plot):
        if metric in df.columns:
            ax = axes[idx // 2, idx % 2]
            df[metric].plot(kind='bar', ax=ax, title=metric.capitalize())
            ax.set_ylabel(metric.capitalize())
            ax.set_xlabel("Model")
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info("Saved comparison plot to %s", output_path)
    plt.close()


__all__ = [
    "evaluate_all_models",
    "compare_models",
    "plot_model_comparison",
]

