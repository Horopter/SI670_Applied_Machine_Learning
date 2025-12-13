"""
XGBoost model using features extracted from pretrained models.

This model uses pretrained models (I3D, R2+1D, ViT, etc.) as feature extractors,
then trains XGBoost on the extracted features. This is memory-efficient and
leverages transfer learning without training the full model from scratch.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import joblib

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Lazy import to avoid circular dependency issues
# VideoConfig and VideoDataset will be imported when needed
from lib.training.feature_preprocessing import remove_collinear_features
from lib.training.model_factory import create_model, get_model_config
from lib.utils.memory import aggressive_gc

logger = logging.getLogger(__name__)


def extract_features_from_pretrained_model(
    model: nn.Module,
    model_type: str,
    dataset,  # VideoDataset - imported lazily to avoid circular dependency
    device: str,
    batch_size: int = 1,
    project_root: str = ""
) -> np.ndarray:
    """
    Extract features from a pretrained model before the final classification layer.
    
    Args:
        model: Pretrained PyTorch model
        model_type: Model type identifier (for feature extraction strategy)
        dataset: Video dataset
        device: Device to run model on
        batch_size: Batch size for feature extraction
        project_root: Project root directory
    
    Returns:
        Extracted features as numpy array (n_samples, feature_dim)
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("xgboost is required. Install with: pip install xgboost")
    
    model.eval()
    model = model.to(device)
    
    # CRITICAL: Use variable_ar_collate to convert (N, T, C, H, W) to (N, C, T, H, W) for 3D CNNs
    # VideoDataset returns clips as (T, C, H, W), which becomes (N, T, C, H, W) when batched
    # 3D CNN models (I3D, R2+1D, PretrainedInception, X3D, SlowFast) expect (N, C, T, H, W)
    try:
        from lib.models.video import variable_ar_collate
        use_collate = True
    except ImportError:
        logger.warning("variable_ar_collate not available, will manually permute clips")
        use_collate = False
    
    # CRITICAL: Set PyTorch memory optimizations before feature extraction
    device_obj = torch.device(device) if isinstance(device, str) else device
    if device_obj.type == "cuda":
        if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch.backends.cudnn.benchmark = False
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        aggressive_gc(clear_cuda=True)
        logger.info("Applied PyTorch memory optimizations for feature extraction")
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        collate_fn=variable_ar_collate if use_collate else None
    )
    
    all_features = []
    
    with torch.no_grad():
        for batch_idx, (clips, _) in enumerate(loader):
            clips = clips.to(device, non_blocking=True)
            
            # CRITICAL: Handle input format conversion based on model type
            # VideoDataset returns clips as (T, C, H, W), which becomes (N, T, C, H, W) when batched
            # 3D CNN models (I3D, R2+1D, PretrainedInception, X3D, SlowFast) expect (N, C, T, H, W)
            # ViT models expect (N, T, C, H, W) which is reshaped to (N*T, C, H, W)
            if clips.dim() == 5:
                # Determine current format
                N, dim1, dim2, H, W = clips.shape
                is_ncthw = (dim1 == 3)  # (N, C, T, H, W) if dim1 == 3
                is_ntchw = (dim2 == 3 and dim1 != 3)  # (N, T, C, H, W) if dim2 == 3 and dim1 != 3
                
                if model_type in ["vit_gru", "vit_transformer"]:
                    # ViT models need (N, T, C, H, W) format
                    if is_ncthw:
                        # Convert from (N, C, T, H, W) to (N, T, C, H, W)
                        clips = clips.permute(0, 2, 1, 3, 4).contiguous()
                        logger.debug(f"Permuted clips from (N, C, T, H, W) to (N, T, C, H, W) for {model_type}")
                    elif not is_ntchw:
                        logger.warning(
                            f"Unexpected clip shape for {model_type}: {clips.shape}. "
                            f"Expected (N, C, T, H, W) or (N, T, C, H, W). Attempting to proceed..."
                        )
                else:
                    # 3D CNN models need (N, C, T, H, W) format
                    if is_ntchw:
                        # Convert from (N, T, C, H, W) to (N, C, T, H, W)
                        clips = clips.permute(0, 2, 1, 3, 4).contiguous()
                        logger.debug(f"Permuted clips from (N, T, C, H, W) to (N, C, T, H, W) for {model_type}")
                    elif not is_ncthw:
                        logger.warning(
                            f"Unexpected clip shape for {model_type}: {clips.shape}. "
                            f"Expected (N, C, T, H, W) or (N, T, C, H, W). Attempting to proceed..."
                        )
            
            # Extract features based on model type
            # CRITICAL: Wrap in try-except to handle OOM errors gracefully
            try:
                if model_type in ["i3d", "r2plus1d", "pretrained_inception"]:
                    # For I3D, R2+1D, PretrainedInceptionVideoModel
                    # Extract features before final fc layer
                    if hasattr(model, 'backbone'):
                        # I3D, R2+1D
                        x = model.backbone.stem(clips)
                        x = model.backbone.layer1(x)
                        x = model.backbone.layer2(x)
                        x = model.backbone.layer3(x)
                        x = model.backbone.layer4(x)
                        
                        # Global average pooling
                        if hasattr(model.backbone, 'avgpool'):
                            x = model.backbone.avgpool(x)
                        else:
                            # Adaptive pooling if no avgpool
                            x = nn.AdaptiveAvgPool3d((1, 1, 1))(x)
                        
                        features = torch.flatten(x, 1)  # (N, feature_dim)
                    elif hasattr(model, 'stem') and hasattr(model, 'layer4'):
                        # PretrainedInceptionVideoModel
                        x = model.stem(clips)
                        x = model.layer1(x)
                        x = model.layer2(x)
                        x = model.layer3(x)
                        x = model.layer4(x)
                        x = model.incept(x)
                        x = model.pool(x)  # AdaptiveAvgPool3d
                        features = torch.flatten(x, 1)  # (N, feature_dim)
                    else:
                        # Fallback: try to get features from model
                        # Remove final layer and extract
                        raise ValueError(f"Cannot extract features from {model_type}: unknown architecture")
                
                elif model_type in ["vit_gru", "vit_transformer"]:
                    # For ViT models, use forward_features
                    # CRITICAL: Clips should already be in (N, T, C, H, W) format from the shape conversion above
                    if hasattr(model, 'vit_backbone'):
                        if clips.dim() == 5:
                            N, T, C, H, W = clips.shape
                            # Reshape to (N*T, C, H, W) for ViT processing
                            clips = clips.view(N * T, C, H, W)
                        else:
                            # Fallback for unexpected dimensions
                            N = clips.shape[0]
                            T = 1
                            clips = clips.view(N * T, -1)
                        
                        # Extract features using ViT
                        vit_output = model.vit_backbone.forward_features(clips)
                        # vit_output shape: (N*T, num_patches+1, embed_dim)
                        # Extract [CLS] token
                        frame_features = vit_output[:, 0, :]  # (N*T, embed_dim)
                        
                        # Reshape back to (N, T, embed_dim) if temporal
                        if T > 1:
                            frame_features = frame_features.view(N, T, -1)
                            # Mean pool over temporal dimension
                            features = frame_features.mean(dim=1)  # (N, embed_dim)
                        else:
                            features = frame_features  # (N, embed_dim)
                    else:
                        raise ValueError(f"Cannot extract features from {model_type}: no vit_backbone")
                
                elif model_type in ["slowfast", "x3d"]:
                    # For SlowFast, X3D - extract before final fc
                    # CRITICAL: These models expect (N, C, T, H, W) format
                    # The shape conversion above should have already converted clips to correct format
                    if hasattr(model, 'backbone'):
                        # Run through backbone but stop before final fc
                        # Most torchvision video models have: stem -> layers -> avgpool -> fc
                        # We need to extract after avgpool but before fc
                        x = model.backbone.stem(clips)
                        x = model.backbone.layer1(x)
                        x = model.backbone.layer2(x)
                        x = model.backbone.layer3(x)
                        x = model.backbone.layer4(x)
                        
                        # Global average pooling
                        if hasattr(model.backbone, 'avgpool'):
                            x = model.backbone.avgpool(x)
                        else:
                            x = nn.AdaptiveAvgPool3d((1, 1, 1))(x)
                        
                        features = torch.flatten(x, 1)  # (N, feature_dim)
                    else:
                        raise ValueError(f"Cannot extract features from {model_type}: no backbone")
                
                else:
                    # Generic: try to extract features by removing final layer
                    # This is a fallback and may not work for all models
                    logger.warning(f"Using generic feature extraction for {model_type}")
                    # Try to get features by running through model without final layer
                    # This is model-specific and may need customization
                    raise NotImplementedError(f"Feature extraction for {model_type} not implemented")
                
            except RuntimeError as e:
                error_msg = str(e)
                if "out of memory" in error_msg.lower():
                    logger.error(
                        f"CUDA OOM during feature extraction (batch {batch_idx + 1}): {e}. "
                        f"Model: {model_type}, Batch size: {batch_size}. "
                        f"GPU memory may be insufficient. Try reducing batch_size or num_frames."
                    )
                    # Clear cache and raise
                    if device_obj.type == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    aggressive_gc(clear_cuda=device_obj.type == "cuda")
                    raise RuntimeError(
                        f"CUDA OOM during feature extraction. "
                        f"Try reducing batch_size (current: {batch_size}) or num_frames (current: {dataset.config.num_frames if hasattr(dataset, 'config') else 'unknown'})."
                    ) from e
                else:
                    # Re-raise non-OOM errors
                    raise
            
            # Convert to numpy and collect
            features_np = features.cpu().numpy()
            all_features.append(features_np)
            
            # CRITICAL: Aggressive memory cleanup after each batch to prevent OOM
            del clips, features, features_np
            if device_obj.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            aggressive_gc(clear_cuda=device_obj.type == "cuda")
            
            # Log progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                logger.debug(f"Processed {batch_idx + 1} batches for feature extraction")
                if device_obj.type == "cuda":
                    logger.debug(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # Concatenate all features
    all_features = np.vstack(all_features)
    logger.info(f"Extracted features shape: {all_features.shape}")
    
    # Final memory cleanup
    if device_obj.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    aggressive_gc(clear_cuda=device_obj.type == "cuda")
    
    return all_features


class XGBoostPretrainedBaseline:
    """
    XGBoost model using features extracted from pretrained models.
    
    Uses pretrained models (I3D, R2+1D, ViT, etc.) as feature extractors,
    then trains XGBoost on the extracted features.
    """
    
    def __init__(
        self,
        base_model_type: str = "i3d",
        cache_dir: Optional[str] = None,
        num_frames: int = 1000,
        xgb_params: Optional[Dict] = None
    ):
        """
        Initialize XGBoost model with pretrained feature extractor.
        
        Args:
            base_model_type: Pretrained model to use as feature extractor
                            (e.g., "i3d", "r2plus1d", "vit_gru", "pretrained_inception")
            cache_dir: Directory to cache extracted features
            num_frames: Number of frames per video
            xgb_params: XGBoost parameters (default: optimized for binary classification)
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost is required. Install with: pip install xgboost")
        
        self.base_model_type = base_model_type
        self.cache_dir = cache_dir
        self.num_frames = num_frames
        self.base_model = None  # Will be loaded when needed
        self.model = None
        self.is_fitted = False
        self.feature_indices = None
        self.feature_names = None
        
        # Default XGBoost parameters optimized for binary classification
        self.xgb_params = xgb_params or {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'tree_method': 'hist',  # Memory-efficient
            'n_jobs': 1,  # Conservative for memory
        }
    
    def _get_cache_path(self, video_path: str, project_root: str) -> Optional[Path]:
        """Get cache path for extracted features."""
        if not self.cache_dir:
            return None
        
        cache_path = Path(project_root) / self.cache_dir
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Create hash from video path
        import hashlib
        video_hash = hashlib.md5(video_path.encode()).hexdigest()
        cache_file = cache_path / f"{self.base_model_type}_{video_hash}_{self.num_frames}.npy"
        
        return cache_file
    
    def _extract_features_batch(
        self,
        video_paths: List[str],
        labels: List[int],
        project_root: str,
        device: str = "cuda"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from pretrained model for a batch of videos.
        
        Args:
            video_paths: List of video paths
            labels: List of labels
            project_root: Project root directory
            device: Device to run model on
        
        Returns:
            Tuple of (features, labels) as numpy arrays
        """
        # Load or create base model
        if self.base_model is None:
            # CRITICAL: Clear GPU memory before loading model to prevent OOM
            device_obj = torch.device(device) if isinstance(device, str) else device
            if device_obj.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                aggressive_gc(clear_cuda=True)
            
            logger.info(f"Loading pretrained model: {self.base_model_type} (num_frames={self.num_frames})")
            model_config = get_model_config(self.base_model_type)
            model_config["num_frames"] = self.num_frames
            self.base_model = create_model(self.base_model_type, model_config)
            
            # Freeze entire model (feature extractor only)
            for param in self.base_model.parameters():
                param.requires_grad = False
            self.base_model.eval()
            
            # CRITICAL: Move model to device and clear cache after loading
            if device_obj.type == "cuda":
                self.base_model = self.base_model.to(device_obj)
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info(f"Loaded {self.base_model_type} model to GPU, cleared cache")
        
        # Create dataset - lazy import to avoid circular dependency
        from lib.models import VideoConfig, VideoDataset
        
        df = pl.DataFrame({
            "video_path": video_paths,
            "label": labels
        })
        
        # Handle both old and new VideoConfig versions (some servers may not have fixed_size parameter)
        # CRITICAL: Stage 5 ALWAYS uses scaled videos from Stage 3, so use_scaled_videos must be True
        try:
            video_config = VideoConfig(
                num_frames=self.num_frames,
                fixed_size=256,
                use_scaled_videos=True  # Stage 5 always uses scaled videos
            )
        except TypeError:
            # Fallback: server version doesn't support these parameters
            logger.warning(
                "VideoConfig on server doesn't support 'fixed_size' or 'use_scaled_videos' parameters. "
                "Using default VideoConfig and setting manually."
            )
            video_config = VideoConfig(num_frames=self.num_frames)
            # CRITICAL: Set use_scaled_videos=True even if constructor doesn't support it
            video_config.use_scaled_videos = True
            logger.info("Manually set use_scaled_videos=True on VideoConfig (server version fallback)")
        
        # CRITICAL: Verify use_scaled_videos is True (Stage 5 requirement)
        if not getattr(video_config, 'use_scaled_videos', False):
            logger.warning(
                "CRITICAL: use_scaled_videos is False in VideoConfig for XGBoost feature extraction! "
                "Stage 5 ALWAYS uses scaled videos from Stage 3. Forcing use_scaled_videos=True."
            )
            video_config.use_scaled_videos = True
            logger.info("Forced use_scaled_videos=True on VideoConfig (Stage 5 requirement)")
        
        dataset = VideoDataset(df, project_root=project_root, config=video_config, train=False)
        
        # CRITICAL: Clear GPU memory before feature extraction
        device_obj = torch.device(device) if isinstance(device, str) else device
        if device_obj.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            aggressive_gc(clear_cuda=True)
        
        # Extract features
        logger.info(f"Extracting features using {self.base_model_type} (num_frames={self.num_frames})...")
        features = extract_features_from_pretrained_model(
            self.base_model,
            self.base_model_type,
            dataset,
            device=device,
            batch_size=1,  # Conservative for memory
            project_root=project_root
        )
        
        # CRITICAL: Clear base model from GPU after feature extraction to free memory
        if device_obj.type == "cuda" and self.base_model is not None:
            self.base_model = self.base_model.cpu()
            del self.base_model
            self.base_model = None  # Force reload next time if needed
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            aggressive_gc(clear_cuda=True)
            logger.info(f"Cleared {self.base_model_type} model from GPU after feature extraction")
        
        labels_array = np.array(labels)
        
        return features, labels_array
    
    def fit(self, df: pl.DataFrame, project_root: str) -> None:
        """
        Train the XGBoost model on features extracted from pretrained model.
        
        Args:
            df: DataFrame with video_path and label columns
            project_root: Project root directory
        """
        logger.info(f"Training XGBoost on features from {self.base_model_type}...")
        logger.info(f"Processing {df.height} videos...")
        
        video_paths = df["video_path"].to_list()
        labels = df["label"].to_list()
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Extract features
        features, y = self._extract_features_batch(
            video_paths,
            labels,
            project_root,
            device=device
        )
        
        # Aggressive GC after feature extraction
        aggressive_gc(clear_cuda=True)
        
        # Get feature names
        feature_names = [f"{self.base_model_type}_feature_{i}" for i in range(features.shape[1])]
        
        # Remove collinear features
        logger.info("Removing collinear features...")
        features_filtered, self.feature_indices, self.feature_names = remove_collinear_features(
            features,
            feature_names=feature_names,
            correlation_threshold=0.95,
            method="correlation"
        )
        logger.info(f"Using {len(self.feature_names)} features after collinearity removal")
        
        # Convert labels to binary (0/1)
        label_map = {label: idx for idx, label in enumerate(sorted(set(labels)))}
        y_binary = np.array([label_map[label] for label in labels])
        
        # Train XGBoost
        logger.info("Training XGBoost...")
        self.model = xgb.XGBClassifier(**self.xgb_params)
        self.model.fit(features_filtered, y_binary)
        self.is_fitted = True
        
        logger.info("✓ XGBoost trained on pretrained model features")
    
    def predict(self, df: pl.DataFrame, project_root: str) -> np.ndarray:
        """
        Predict labels for videos.
        
        Args:
            df: DataFrame with video_path column
            project_root: Project root directory
        
        Returns:
            Probability array (n_samples, 2) for binary classification
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        video_paths = df["video_path"].to_list()
        labels = df["label"].to_list() if "label" in df.columns else [0] * len(video_paths)
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Extract features
        features, _ = self._extract_features_batch(
            video_paths,
            labels,
            project_root,
            device=device
        )
        
        # Filter features using stored indices
        if self.feature_indices is not None:
            features_filtered = features[:, self.feature_indices]
        else:
            features_filtered = features
        
        # Predict probabilities
        probs_positive = self.model.predict_proba(features_filtered)[:, 1]  # Probability of class 1
        
        # Return as (n_samples, 2) array
        probs = np.column_stack([1 - probs_positive, probs_positive])
        
        return probs
    
    def save(self, path: str) -> None:
        """Save model and scaler to disk."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Cannot save.")
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost model
        if self.model is not None:
            self.model.save_model(str(save_path / "xgboost_model.json"))
        
        # Save metadata
        import json
        metadata = {
            "base_model_type": self.base_model_type,
            "num_frames": self.num_frames,
            "xgb_params": self.xgb_params,
            "feature_indices": self.feature_indices,
            "feature_names": self.feature_names,
        }
        with open(save_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved XGBoost model to {save_path}")
    
    def load(self, path: str) -> None:
        """Load model and scaler from disk."""
        load_path = Path(path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model path not found: {load_path}")
        
        # Load XGBoost model
        model_file = load_path / "xgboost_model.json"
        if model_file.exists():
            self.model = xgb.XGBClassifier()
            self.model.load_model(str(model_file))
        else:
            raise FileNotFoundError(f"XGBoost model file not found: {model_file}")
        
        # Load metadata
        metadata_file = load_path / "metadata.json"
        if metadata_file.exists():
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            self.base_model_type = metadata["base_model_type"]
            self.num_frames = metadata["num_frames"]
            self.xgb_params = metadata.get("xgb_params", {})
            self.feature_indices = metadata.get("feature_indices")
            self.feature_names = metadata.get("feature_names")
        
        self.is_fitted = True
        logger.info(f"Loaded XGBoost model from {load_path}")


__all__ = ["XGBoostPretrainedBaseline", "extract_features_from_pretrained_model"]

