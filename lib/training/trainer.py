"""
Core training utilities.

Provides:
- Optimizer and scheduler builders
- Training and evaluation loops
- Configuration dataclasses
- Early stopping
- Model freezing utilities
- Class weight computation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Iterable
import torch
import torch.nn as nn
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import _LRScheduler, StepLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from lib.models import VariableARVideoModel

logger = logging.getLogger(__name__)


@dataclass
class OptimConfig:
    """Optimizer configuration."""
    lr: float = 1e-4
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)


@dataclass
class TrainConfig:
    """Training configuration."""
    num_epochs: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_interval: int = 10
    use_class_weights: bool = True
    use_amp: bool = True
    checkpoint_dir: Optional[str] = None
    early_stopping_patience: int = 5
    gradient_accumulation_steps: int = 1


class EarlyStopping:
    """Early stops the training if validation metric doesn't improve."""
    
    def __init__(self, patience: int = 5, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode  # "max" for metrics like accuracy/F1, "min" for loss
        self.best: Optional[float] = None
        self.counter = 0
        self.should_stop = False
    
    def step(self, value: float) -> None:
        if self.best is None:
            self.best = value
            self.counter = 0
            return
        
        improved = value > self.best if self.mode == "max" else value < self.best
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def freeze_all(model: nn.Module) -> None:
    """Freeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = True


def freeze_backbone_unfreeze_head(model: VariableARVideoModel) -> None:
    """Freeze backbone but unfreeze head."""
    if hasattr(model, 'backbone'):
        freeze_all(model.backbone)
    if hasattr(model, 'head'):
        unfreeze_all(model.head)
    elif hasattr(model, 'classifier'):
        unfreeze_all(model.classifier)


def trainable_params(model: nn.Module) -> Iterable[torch.nn.Parameter]:
    """Get trainable parameters."""
    return (p for p in model.parameters() if p.requires_grad)


def compute_class_counts(loader: DataLoader, num_classes: int) -> torch.Tensor:
    """Compute class counts from a DataLoader."""
    counts = torch.zeros(num_classes, dtype=torch.long)
    for _, labels in loader:
        for label in labels:
            counts[label.item()] += 1
    return counts


def make_class_weights(counts: torch.Tensor) -> torch.Tensor:
    """Make class weights from counts (inverse frequency)."""
    total = counts.sum().float()
    weights = total / (counts.float() + 1e-6)  # Add epsilon to avoid division by zero
    weights = weights / weights.sum() * len(weights)  # Normalize
    return weights


def make_weighted_sampler(labels: torch.Tensor) -> WeightedRandomSampler:
    """Make weighted random sampler from labels."""
    unique_labels = torch.unique(labels)
    counts = torch.bincount(labels)
    weights = make_class_weights(counts)
    sample_weights = weights[labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights))


def build_optimizer(model: nn.Module, config: OptimConfig) -> Optimizer:
    """Build optimizer from config."""
    return Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=config.betas
    )


def build_scheduler(
    optimizer: Optimizer, 
    step_size: int = 10, 
    gamma: float = 0.1
) -> _LRScheduler:
    """Build learning rate scheduler."""
    return StepLR(optimizer, step_size=step_size, gamma=gamma)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    device: str,
    use_class_weights: bool = True,
    use_amp: bool = True,
    epoch: int = 0,
    log_interval: int = 10,
    gradient_accumulation_steps: int = 1,
) -> float:
    """
    Train model for one epoch.
    
    Returns:
        Average training loss
    """
    from lib.utils.memory import aggressive_gc
    
    model.train()
    total_loss = 0.0
    
    # Clear CUDA cache at start
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    
    # Loss criterion (inferred on first batch)
    first_batch = True
    criterion: Optional[nn.Module] = None
    
    # AMP scaler
    if use_amp and device.startswith("cuda"):
        try:
            scaler = torch.amp.GradScaler('cuda')
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    
    for batch_idx, (clips, labels) in enumerate(loader):
        # Aggressive GC every 5 batches
        if batch_idx > 0 and batch_idx % 5 == 0:
            import gc
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        
        clips = clips.to(device)
        labels = labels.to(device)
        
        # Zero gradients at start of accumulation cycle
        if batch_idx % gradient_accumulation_steps == 0:
            optimizer.zero_grad()
        
        # Infer criterion on first batch
        if first_batch:
            with torch.no_grad():
                logits = model(clips)
                if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
                    criterion = nn.BCEWithLogitsLoss()
                else:
                    criterion = nn.CrossEntropyLoss()
            first_batch = False
        
        # Forward pass with AMP
        if scaler is not None:
            try:
                with torch.amp.autocast('cuda'):
                    logits = model(clips)
            except (AttributeError, TypeError):
                with torch.cuda.amp.autocast():
                    logits = model(clips)
        else:
            logits = model(clips)
        
        # Compute loss
        if isinstance(criterion, nn.BCEWithLogitsLoss):
            if logits.ndim == 2 and logits.shape[1] == 1:
                logits = logits.squeeze(-1)
            targets = labels.float()
            loss = criterion(logits, targets)
        else:
            if logits.ndim == 1:
                logits = logits.unsqueeze(-1)
            loss = criterion(logits, labels)
        
        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights at end of accumulation cycle
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            aggressive_gc(clear_cuda=device.startswith("cuda"))
        
        total_loss += float(loss.item() * gradient_accumulation_steps)
        
        # Logging
        if (batch_idx + 1) % log_interval == 0:
            logger.info(
                f"Epoch {epoch}, Batch {batch_idx + 1}/{len(loader)}, "
                f"Loss: {loss.item() * gradient_accumulation_steps:.4f}"
            )
    
    avg_loss = total_loss / max(1, len(loader))
    return avg_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> Tuple[float, float]:
    """
    Evaluate model on validation/test set.
    
    Returns:
        (average_loss, accuracy)
    """
    from lib.utils.memory import aggressive_gc
    
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    criterion: Optional[nn.Module] = None
    first_batch = True
    
    for clips, labels in loader:
        clips = clips.to(device)
        labels = labels.to(device)
        
        # Use autocast for mixed precision
        if device.startswith("cuda"):
            try:
                with torch.amp.autocast('cuda'):
                    logits = model(clips)
            except (AttributeError, TypeError):
                with torch.cuda.amp.autocast():
                    logits = model(clips)
        else:
            logits = model(clips)
        
        # Infer criterion on first batch
        if first_batch:
            if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
                if logits.ndim == 2:
                    logits = logits.squeeze(-1)
                criterion = nn.BCEWithLogitsLoss()
            else:
                criterion = nn.CrossEntropyLoss()
            first_batch = False
        
        # Compute loss and predictions
        if isinstance(criterion, nn.BCEWithLogitsLoss):
            if logits.ndim == 2 and logits.shape[1] == 1:
                logits = logits.squeeze(-1)
            targets = labels.float()
            loss = criterion(logits, targets)
            preds = (logits.sigmoid() >= 0.5).long()
        else:
            if logits.ndim == 1:
                logits = logits.unsqueeze(-1)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
        
        total_correct += int((preds == labels).sum().item())
        total_samples += int(labels.numel())
        total_loss += float(loss.item())
        
        # Clear intermediate tensors
        del logits, preds, loss, clips, labels
        aggressive_gc(clear_cuda=device.startswith("cuda"))
    
    avg_loss = total_loss / max(1, len(loader))
    acc = total_correct / max(1, total_samples)
    
    # Final aggressive GC
    aggressive_gc(clear_cuda=device.startswith("cuda"))
    
    return avg_loss, acc


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optim_cfg: OptimConfig,
    train_cfg: TrainConfig,
) -> nn.Module:
    """
    High-level training loop with optional validation.
    
    Returns:
        Trained model
    """
    device = train_cfg.device
    model.to(device)
    model.train()
    
    optimizer = build_optimizer(model, optim_cfg)
    scheduler = build_scheduler(optimizer)
    
    best_val_acc = 0.0
    
    for epoch in range(1, train_cfg.num_epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device=device,
            use_class_weights=train_cfg.use_class_weights,
            use_amp=train_cfg.use_amp,
            epoch=epoch,
            log_interval=train_cfg.log_interval,
            gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        )
        
        logger.info(f"Epoch {epoch}/{train_cfg.num_epochs}, Train Loss: {train_loss:.4f}")
        
        # Validate
        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, device=device)
            logger.info(f"Epoch {epoch}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                logger.info(f"New best validation accuracy: {best_val_acc:.4f}")
        
        # Step scheduler
        scheduler.step()
    
    return model

