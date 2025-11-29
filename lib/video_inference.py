"""
Inference utilities for the FVC video classifier.

Provides a simple API:
- load_model_from_checkpoint(...)
- predict_video(...)
- batch_predict_from_csv(...)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import os
import polars as pl
import torch

from .video_modeling import VideoConfig, VideoDataset, variable_ar_collate, VariableARVideoModel, PretrainedInceptionVideoModel
from .video_data import load_metadata


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: Optional[str] = None,
    model_type: str = "pretrained",  # "pretrained" or "variable_ar"
) -> torch.nn.Module:
    """Load model weights from a checkpoint path.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Target device (auto-detected if None)
        model_type: "pretrained" for PretrainedInceptionVideoModel, 
                   "variable_ar" for VariableARVideoModel
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if model_type == "pretrained":
        model = PretrainedInceptionVideoModel(freeze_backbone=False)
    else:
        model = VariableARVideoModel()
    
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_video(
    model: torch.nn.Module,  # Accept any model type
    video_path: str,
    project_root: str,
    config: Optional[VideoConfig] = None,
    device: Optional[str] = None,
    use_rolling_window: bool = False,
    aggregation: str = "mean",  # "mean", "max", "median", "vote"
) -> Dict[str, float]:
    """
    Run prediction on a single video path.
    
    Args:
        model: Trained model
        video_path: Path to video file
        project_root: Project root directory
        config: VideoConfig (if None, uses defaults)
        device: Target device
        use_rolling_window: If True, use rolling windows and aggregate predictions
        aggregation: How to aggregate rolling window predictions ("mean", "max", "median", "vote")
    
    Returns:
        Dictionary with "logit" and "prob" keys
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if config is None:
        config = VideoConfig()
    
    # If using rolling windows, aggregate predictions from multiple windows
    if use_rolling_window:
        return predict_video_rolling_window(
            model, video_path, project_root, config, device, aggregation
        )
    
    # Standard single-window prediction
    # Build a temporary single-row Polars DataFrame for compatibility with VideoDataset
    df = pl.DataFrame({
        "video_path": [video_path],
        "label": [0],  # dummy label, not used
    })

    dataset = VideoDataset(df, project_root, config=config, train=False)
    clip, _ = dataset[0]
    # (T, C, H, W) -> (1, C, T, H, W)
    clip = clip.unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)

    logits = model(clip).squeeze(-1)
    prob = float(logits.sigmoid().item())
    return {"logit": float(logits.item()), "prob": prob}


@torch.no_grad()
def predict_video_rolling_window(
    model: torch.nn.Module,
    video_path: str,
    project_root: str,
    config: VideoConfig,
    device: str,
    aggregation: str = "mean",
) -> Dict[str, float]:
    """
    Predict using rolling windows and aggregate results.
    
    This is more robust for detecting real vs fake across the entire video,
    as it analyzes multiple temporal segments and combines their predictions.
    """
    from .video_modeling import _read_video_wrapper, rolling_window_indices, build_frame_transforms
    import numpy as np
    
    # Read video
    video = _read_video_wrapper(video_path)
    if video.shape[0] == 0:
        raise RuntimeError(f"Video has no frames: {video_path}")
    
    total_frames = video.shape[0]
    window_size = config.window_size or config.num_frames
    stride = config.window_stride or (window_size // 2)
    
    # Generate rolling windows
    windows = rolling_window_indices(total_frames, window_size, stride)
    if not windows:
        # Fallback to single window
        windows = [list(range(min(window_size, total_frames)))]
    
    # Process each window
    frame_transform = build_frame_transforms(train=False)
    all_logits = []
    
    for window_indices in windows:
        frames = []
        for i in window_indices:
            frame = video[i].numpy().astype(np.uint8)  # (H, W, C)
            frame_tensor = frame_transform(frame)  # (C, H, W)
            frames.append(frame_tensor)
        
        clip = torch.stack(frames, dim=0)  # (T, C, H, W)
        clip = clip.unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)  # (1, C, T, H, W)
        
        logits = model(clip).squeeze(-1)  # (1,)
        all_logits.append(float(logits.item()))
    
    # Aggregate predictions
    all_logits_tensor = torch.tensor(all_logits)
    
    if aggregation == "mean":
        final_logit = float(all_logits_tensor.mean().item())
    elif aggregation == "max":
        final_logit = float(all_logits_tensor.max().item())
    elif aggregation == "median":
        final_logit = float(torch.median(all_logits_tensor).item())
    elif aggregation == "vote":
        # Vote based on binary predictions
        probs = all_logits_tensor.sigmoid()
        votes = (probs >= 0.5).float()
        final_logit = float((votes.mean() - 0.5) * 10.0)  # Convert vote ratio to logit scale
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    
    final_prob = float(torch.sigmoid(torch.tensor(final_logit)).item())
    
    return {
        "logit": final_logit,
        "prob": final_prob,
        "num_windows": len(windows),
        "window_logits": all_logits,  # For debugging/analysis
    }


@torch.no_grad()
def batch_predict_from_csv(
    checkpoint_path: str,
    csv_path: str,
    project_root: str,
    output_csv: str,
    config: Optional[VideoConfig] = None,
    batch_size: int = 4,
) -> None:
    """Run inference over all videos listed in csv_path and write predictions."""
    if config is None:
        config = VideoConfig()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model_from_checkpoint(checkpoint_path, device=device)

    df = load_metadata(csv_path)  # Returns Polars DataFrame
    dataset = VideoDataset(df, project_root, config=config, train=False)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=variable_ar_collate,
    )

    all_probs: List[float] = []
    model.eval()
    for clips, _ in loader:
        clips = clips.to(device)
        logits = model(clips).squeeze(-1)
        probs = logits.sigmoid().cpu().tolist()
        all_probs.extend(probs)

    # Add predictions to Polars DataFrame and write as CSV
    df_out = df.with_columns([
        pl.Series("pred_prob", all_probs),
        pl.Series("pred_label", [(1 if p >= 0.5 else 0) for p in all_probs])
    ])
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_out.write_csv(output_csv)


__all__ = ["load_model_from_checkpoint", "predict_video", "batch_predict_from_csv"]


