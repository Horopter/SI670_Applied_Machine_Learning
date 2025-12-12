"""
Shared utility functions for computing classification metrics.

Eliminates duplicate metric computation code across training scripts.
"""

from __future__ import annotations

from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, precision_recall_fscore_support
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray | None = None
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics for binary classification.
    
    Args:
        y_true: True labels (1D array of 0/1)
        y_pred: Predicted labels (1D array of 0/1)
        y_probs: Predicted probabilities (1D array, optional, for loss computation)
    
    Returns:
        Dictionary with metrics:
        {
            "val_loss": float,
            "val_acc": float,
            "val_f1": float,
            "val_precision": float,
            "val_recall": float,
            "val_f1_class0": float,
            "val_precision_class0": float,
            "val_recall_class0": float,
            "val_f1_class1": float,
            "val_precision_class1": float,
            "val_recall_class1": float,
        }
    """
    # Overall metrics
    val_acc = float((y_pred == y_true).mean())
    val_precision = float(precision_score(y_true, y_pred, average='binary', zero_division=0))
    val_recall = float(recall_score(y_true, y_pred, average='binary', zero_division=0))
    val_f1 = float(f1_score(y_true, y_pred, average='binary', zero_division=0))
    
    # Compute loss if probabilities provided
    if y_probs is not None:
        # Ensure y_probs is 2D for multi-class case
        if y_probs.ndim == 1:
            # Binary case: convert to 2D (N, 2)
            y_probs_2d = np.stack([1 - y_probs, y_probs], axis=1)
        else:
            y_probs_2d = y_probs
        val_loss = float(-np.mean(np.log(y_probs_2d[np.arange(len(y_true)), y_true] + 1e-10)))
    else:
        val_loss = float('nan')
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    return {
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_f1": val_f1,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_f1_class0": float(f1_per_class[0]) if len(f1_per_class) > 0 else 0.0,
        "val_precision_class0": float(precision_per_class[0]) if len(precision_per_class) > 0 else 0.0,
        "val_recall_class0": float(recall_per_class[0]) if len(recall_per_class) > 0 else 0.0,
        "val_f1_class1": float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0,
        "val_precision_class1": float(precision_per_class[1]) if len(precision_per_class) > 1 else 0.0,
        "val_recall_class1": float(recall_per_class[1]) if len(recall_per_class) > 1 else 0.0,
    }
