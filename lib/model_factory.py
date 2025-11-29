"""
Model factory and registry for centralized model creation.

Supports:
- Baseline models: logistic_regression, svm, naive_cnn
- Frame→temporal: vit_gru, vit_transformer
- Spatiotemporal: slowfast, x3d
- Existing: pretrained_inception
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List
import torch
import torch.nn as nn

from .mlops_core import RunConfig

logger = logging.getLogger(__name__)

# Memory-optimized configurations for each model type
# NOTE: Baseline models use slightly larger batch sizes but remain conservative
# to avoid excessive memory usage during feature extraction.
MODEL_MEMORY_CONFIGS = {
    "logistic_regression": {
        "batch_size": 32,  # Reduced from 64 for OOM safety
        "num_workers": 2,
        "num_frames": 8,
        "gradient_accumulation_steps": 1,
    },
    "svm": {
        "batch_size": 32,  # Reduced from 64 for OOM safety
        "num_workers": 2,
        "num_frames": 8,
        "gradient_accumulation_steps": 1,
    },
    "naive_cnn": {
        "batch_size": 16,
        "num_workers": 2,
        "num_frames": 8,
        "gradient_accumulation_steps": 1,
    },
    "vit_gru": {
        "batch_size": 4,
        "num_workers": 1,
        "num_frames": 8,
        "gradient_accumulation_steps": 4,
    },
    "vit_transformer": {
        "batch_size": 2,
        "num_workers": 1,
        "num_frames": 8,
        "gradient_accumulation_steps": 8,
    },
    "slowfast": {
        "batch_size": 2,
        "num_workers": 1,
        "num_frames": 16,  # SlowFast needs more frames
        "gradient_accumulation_steps": 8,
    },
    "x3d": {
        "batch_size": 4,
        "num_workers": 1,
        "num_frames": 16,
        "gradient_accumulation_steps": 4,
    },
    "pretrained_inception": {
        "batch_size": 8,
        "num_workers": 2,
        "num_frames": 8,
        "gradient_accumulation_steps": 2,
    },
}


def get_model_config(model_type: str) -> Dict[str, Any]:
    """
    Get memory-optimized configuration for a model type.
    
    Args:
        model_type: Model type identifier
    
    Returns:
        Dictionary with batch_size, num_workers, etc.
    """
    if model_type not in MODEL_MEMORY_CONFIGS:
        logger.warning(f"Unknown model type: {model_type}. Using default config.")
        return {
            "batch_size": 8,
            "num_workers": 2,
            "num_frames": 8,
            "gradient_accumulation_steps": 2,
        }
    
    return MODEL_MEMORY_CONFIGS[model_type].copy()


def list_available_models() -> List[str]:
    """List all available model types."""
    return list(MODEL_MEMORY_CONFIGS.keys())


def create_model(model_type: str, config: RunConfig) -> Any:
    """
    Create a model instance based on model type and config.
    
    Args:
        model_type: Model type identifier
        config: RunConfig with model-specific settings
    
    Returns:
        Model instance (PyTorch nn.Module or sklearn-style model)
    """
    # Handle both RunConfig and dict
    if isinstance(config, dict):
        model_specific = config.get("model_specific_config", {})
        num_frames = config.get("num_frames", 8)
    else:
        # RunConfig object - model_specific_config is always a dict
        model_specific = getattr(config, 'model_specific_config', {})
        if not isinstance(model_specific, dict):
            model_specific = {}
        num_frames = getattr(config, 'num_frames', 8)
    
    # Helper to safely get parameter from model_specific dict
    def get_param(key, default):
        if isinstance(model_specific, dict):
            return model_specific.get(key, default)
        return default
    
    if model_type == "logistic_regression":
        from .baseline_models import LogisticRegressionBaseline
        return LogisticRegressionBaseline(
            cache_dir=get_param("feature_cache_dir", None),
            num_frames=get_param("num_frames", num_frames)
        )
    
    elif model_type == "svm":
        from .baseline_models import SVMBaseline
        return SVMBaseline(
            cache_dir=get_param("feature_cache_dir", None),
            num_frames=get_param("num_frames", num_frames)
        )
    
    elif model_type == "naive_cnn":
        from .baseline_models import NaiveCNNBaseline
        return NaiveCNNBaseline(
            num_frames=get_param("num_frames", num_frames),
            num_classes=2
        )
    
    elif model_type == "vit_gru":
        from .frame_temporal_models import ViTGRUModel
        return ViTGRUModel(
            num_frames=get_param("num_frames", num_frames),
            hidden_dim=get_param("hidden_dim", 256),
            num_layers=get_param("num_layers", 2),
            dropout=get_param("dropout", 0.5),
            pretrained=get_param("pretrained", True)
        )
    
    elif model_type == "vit_transformer":
        from .frame_temporal_models import ViTTransformerModel
        return ViTTransformerModel(
            num_frames=get_param("num_frames", num_frames),
            d_model=get_param("d_model", 768),
            nhead=get_param("nhead", 8),
            num_layers=get_param("num_layers", 2),
            dim_feedforward=get_param("dim_feedforward", 2048),
            dropout=get_param("dropout", 0.5),
            pretrained=get_param("pretrained", True)
        )
    
    elif model_type == "slowfast":
        from .spatiotemporal_models import SlowFastModel
        return SlowFastModel(
            slow_frames=get_param("slow_frames", 16),
            fast_frames=get_param("fast_frames", 64),
            alpha=get_param("alpha", 8),
            beta=get_param("beta", 1.0 / 8),
            pretrained=get_param("pretrained", True)
        )
    
    elif model_type == "x3d":
        from .spatiotemporal_models import X3DModel
        return X3DModel(
            variant=get_param("variant", "x3d_m"),
            pretrained=get_param("pretrained", True)
        )
    
    elif model_type == "pretrained_inception":
        from .video_modeling import PretrainedInceptionVideoModel
        return PretrainedInceptionVideoModel(
            freeze_backbone=get_param("freeze_backbone", False)
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list_available_models()}")


def is_pytorch_model(model_type: str) -> bool:
    """
    Check if model type is a PyTorch model (vs sklearn baseline).
    
    Args:
        model_type: Model type identifier
    
    Returns:
        True if PyTorch model, False if sklearn baseline
    """
    sklearn_models = {"logistic_regression", "svm"}
    return model_type not in sklearn_models


def get_model_input_shape(model_type: str, config: RunConfig) -> tuple:
    """
    Get expected input shape for a model.
    
    Args:
        model_type: Model type identifier
        config: RunConfig
    
    Returns:
        Input shape tuple (C, T, H, W) or description
    """
    num_frames = config.num_frames
    fixed_size = config.fixed_size or 224
    
    if model_type in ["logistic_regression", "svm"]:
        return "features"  # Handcrafted features, not video
    
    elif model_type in ["naive_cnn", "vit_gru", "vit_transformer"]:
        return (3, num_frames, fixed_size, fixed_size)
    
    elif model_type in ["slowfast", "x3d", "pretrained_inception"]:
        return (3, num_frames, fixed_size, fixed_size)
    
    else:
        return (3, num_frames, fixed_size, fixed_size)


__all__ = [
    "MODEL_MEMORY_CONFIGS",
    "get_model_config",
    "list_available_models",
    "create_model",
    "is_pytorch_model",
    "get_model_input_shape",
]

