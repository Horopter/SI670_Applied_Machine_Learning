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
from ._linear import LogisticRegressionBaseline
from ._svm import SVMBaseline
from ._cnn import NaiveCNNBaseline
from ._transformer_gru import ViTGRUModel
from ._transformer import ViTTransformerModel
from .slowfast import SlowFastModel
from .x3d import X3DModel
from .i3d import I3DModel
from .r2plus1d import R2Plus1DModel

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

# Feature preprocessing
from .feature_preprocessing import remove_collinear_features, load_and_combine_features

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
    "I3DModel",
    "R2Plus1DModel",
    # Factory
    "create_model",
    "get_model_config",
    "is_pytorch_model",
    "list_available_models",
    "download_pretrained_models",
    "MODEL_MEMORY_CONFIGS",
    # Stage 5
    "stage5_train_models",
    # Feature preprocessing
    "remove_collinear_features",
    "load_and_combine_features",
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

