"""
Model training module.

Provides:
- Baseline models (Logistic Regression, SVM, Naive CNN)
- Frame-temporal models (ViT+GRU, ViT+Transformer)
- Spatiotemporal models (SlowFast, X3D)
- Training utilities and loops
- Model factory
- Stage 5: Training pipeline
"""

# Core training utilities
from .trainer import (
    OptimConfig,
    TrainConfig,
    build_optimizer,
    build_scheduler,
    train_one_epoch,
    evaluate,
    fit,
)

# Models
from .logistic_regression import LogisticRegressionBaseline
from .svm import SVMBaseline
from .naive_cnn import NaiveCNNBaseline
from .vit_gru import ViTGRUModel
from .vit_transformer import ViTTransformerModel
from .slowfast import SlowFastModel
from .x3d import X3DModel

# Model factory
from .model_factory import (
    create_model,
    get_model_config,
    is_pytorch_model,
    list_available_models,
    download_pretrained_models,
    MODEL_MEMORY_CONFIGS,
)

# Training pipeline
from .pipeline import stage5_train_models

# Additional utilities
from .trainer import (
    EarlyStopping,
    freeze_all,
    unfreeze_all,
    freeze_backbone_unfreeze_head,
    trainable_params,
    compute_class_counts,
    make_class_weights,
    make_weighted_sampler,
)

__all__ = [
    # Core
    "OptimConfig",
    "TrainConfig",
    "build_optimizer",
    "build_scheduler",
    "train_one_epoch",
    "evaluate",
    "fit",
    # Models
    "LogisticRegressionBaseline",
    "SVMBaseline",
    "NaiveCNNBaseline",
    "ViTGRUModel",
    "ViTTransformerModel",
    "SlowFastModel",
    "X3DModel",
    # Factory
    "create_model",
    "get_model_config",
    "is_pytorch_model",
    "list_available_models",
    "download_pretrained_models",
    "MODEL_MEMORY_CONFIGS",
    # Stage 5
    "stage5_train_models",
    # Utilities
    "EarlyStopping",
    "freeze_all",
    "unfreeze_all",
    "freeze_backbone_unfreeze_head",
    "trainable_params",
    "compute_class_counts",
    "make_class_weights",
    "make_weighted_sampler",
]

