"""
Visualization utilities for model training results.

Generates graphs for:
- Learning curves (train/val loss and accuracy)
- Confusion matrices
- ROC curves
- Precision-Recall curves
- Cross-validation fold comparison
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, 
    precision_recall_curve, average_precision_score
)

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def plot_learning_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Path,
    train_accs: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None,
    title: str = "Learning Curves"
) -> None:
    """
    Plot learning curves (loss and accuracy over epochs).
    
    Args:
        train_losses: Training loss per epoch
        val_losses: Validation loss per epoch
        train_accs: Training accuracy per epoch (optional)
        val_accs: Validation accuracy per epoch (optional)
        save_path: Path to save the plot
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2 if train_accs is not None else 1, figsize=(15, 5))
    if train_accs is None:
        axes = [axes]
    
    # Plot loss
    ax = axes[0]
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot accuracy if provided
    if train_accs is not None:
        ax = axes[1]
        ax.plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
        ax.plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved learning curves to {save_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Path,
    title: str = "Confusion Matrix"
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
        title: Plot title
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add accuracy text
    accuracy = np.trace(cm) / np.sum(cm)
    ax.text(0.5, -0.15, f'Accuracy: {accuracy:.4f}', 
            transform=ax.transAxes, ha='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved confusion matrix to {save_path}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Path,
    title: str = "ROC Curve"
) -> None:
    """
    Plot ROC curve.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
        save_path: Path to save the plot
        title: Plot title
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved ROC curve to {save_path}")


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Path,
    title: str = "Precision-Recall Curve"
) -> None:
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
        save_path: Path to save the plot
        title: Plot title
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='darkorange', lw=2,
            label=f'PR curve (AP = {avg_precision:.4f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved Precision-Recall curve to {save_path}")


def plot_cv_fold_comparison(
    fold_results: List[Dict[str, Any]],
    save_path: Path,
    title: str = "Cross-Validation Fold Comparison"
) -> None:
    """
    Plot comparison of metrics across CV folds.
    
    Args:
        fold_results: List of results from each fold
        save_path: Path to save the plot
        title: Plot title
    """
    if not fold_results:
        logger.warning("No fold results to plot")
        return
    
    # Extract metrics
    folds = [r.get("fold", i+1) for i, r in enumerate(fold_results)]
    val_accs = [r.get("val_acc", 0) for r in fold_results]
    val_f1s = [r.get("val_f1", 0) for r in fold_results]
    val_precisions = [r.get("val_precision", 0) for r in fold_results]
    val_recalls = [r.get("val_recall", 0) for r in fold_results]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    metrics = [
        (val_accs, "Accuracy", axes[0, 0]),
        (val_f1s, "F1 Score", axes[0, 1]),
        (val_precisions, "Precision", axes[1, 0]),
        (val_recalls, "Recall", axes[1, 1])
    ]
    
    for values, metric_name, ax in metrics:
        ax.bar(folds, values, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Fold')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} by Fold')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for fold, value in zip(folds, values):
            if not np.isnan(value):
                ax.text(fold, value + 0.01, f'{value:.3f}', 
                       ha='center', va='bottom', fontsize=9)
        
        # Add mean line
        mean_val = np.nanmean(values)
        ax.axhline(mean_val, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_val:.3f}')
        ax.legend()
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved CV fold comparison to {save_path}")


def plot_hyperparameter_search(
    search_results: List[Dict[str, Any]],
    save_path: Path,
    title: str = "Hyperparameter Search Results"
) -> None:
    """
    Plot hyperparameter search results.
    
    Args:
        search_results: List of results from grid search
        save_path: Path to save the plot
        title: Plot title
    """
    if not search_results:
        logger.warning("No search results to plot")
        return
    
    # Extract F1 scores and parameter combinations
    f1_scores = [r.get("mean_f1", 0) for r in search_results]
    
    # Create bar plot of top N results
    top_n = min(10, len(search_results))
    sorted_results = sorted(search_results, key=lambda x: x.get("mean_f1", 0), reverse=True)[:top_n]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(top_n)
    scores = [r.get("mean_f1", 0) for r in sorted_results]
    labels = [f"Config {i+1}" for i in range(top_n)]
    
    bars = ax.barh(x_pos, scores, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_yticks(x_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Mean F1 Score')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(score + 0.01, i, f'{score:.4f}', 
               va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved hyperparameter search results to {save_path}")


def generate_all_plots(
    model_type: str,
    fold_results: List[Dict[str, Any]],
    output_dir: Path,
    train_history: Optional[Dict[str, List[float]]] = None,
    predictions: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    class_names: Optional[List[str]] = None
) -> None:
    """
    Generate all plots for a model.
    
    Args:
        model_type: Model type identifier
        fold_results: List of results from each fold
        output_dir: Directory to save plots
        train_history: Training history (train_loss, val_loss, train_acc, val_acc)
        predictions: Tuple of (y_true, y_pred, y_proba) for best fold
        class_names: List of class names
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if class_names is None:
        class_names = ["Real", "Fake"]
    
    # Plot CV fold comparison
    plot_cv_fold_comparison(
        fold_results,
        output_dir / "cv_fold_comparison.png",
        title=f"{model_type} - Cross-Validation Results"
    )
    
    # Plot learning curves if available
    if train_history:
        plot_learning_curves(
            train_history.get("train_loss", []),
            train_history.get("val_loss", []),
            train_history.get("train_acc"),
            train_history.get("val_acc"),
            output_dir / "learning_curves.png",
            title=f"{model_type} - Learning Curves"
        )
    
    # Plot confusion matrix, ROC, PR if predictions available
    if predictions:
        y_true, y_pred, y_proba = predictions
        
        plot_confusion_matrix(
            y_true, y_pred, class_names,
            output_dir / "confusion_matrix.png",
            title=f"{model_type} - Confusion Matrix"
        )
        
        if y_proba is not None and len(y_proba.shape) > 1:
            # Multi-class: use positive class probabilities
            y_proba_pos = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.flatten()
        else:
            y_proba_pos = y_proba
        
        plot_roc_curve(
            y_true, y_proba_pos,
            output_dir / "roc_curve.png",
            title=f"{model_type} - ROC Curve"
        )
        
        plot_precision_recall_curve(
            y_true, y_proba_pos,
            output_dir / "precision_recall_curve.png",
            title=f"{model_type} - Precision-Recall Curve"
        )
    
    logger.info(f"Generated all plots for {model_type} in {output_dir}")

