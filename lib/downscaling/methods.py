"""
Video downscaling methods.

Provides:
- Resolution-based downscaling (letterbox resize)
- Autoencoder-based downscaling (optional)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Try to import torch for autoencoder support
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


def letterbox_resize(
    frame: np.ndarray,
    target_size: int = 224
) -> np.ndarray:
    """
    Resize frame with letterboxing to maintain aspect ratio.
    
    Args:
        frame: Input frame (H, W, 3)
        target_size: Target size for both dimensions
    
    Returns:
        Resized frame (target_size, target_size, 3)
    """
    h, w = frame.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create letterbox
    result = np.zeros((target_size, target_size, 3), dtype=frame.dtype)
    paste_y = (target_size - new_h) // 2
    paste_x = (target_size - new_w) // 2
    result[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = resized
    
    return result


def _autoencoder_downscale(
    frames: list,
    autoencoder: object,
    target_size: int = 224
) -> list:
    """
    Downscale frames using an autoencoder model.
    
    Args:
        frames: List of frames (each is H, W, 3) as numpy arrays
        autoencoder: Autoencoder model (PyTorch nn.Module or object with encode/decode methods)
        target_size: Target size for output frames
    
    Returns:
        List of downscaled frames (target_size, target_size, 3) as numpy arrays
    """
    if not frames:
        return []
    
    if not TORCH_AVAILABLE or torch is None:
        raise RuntimeError("PyTorch is required for autoencoder downscaling")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if autoencoder is a PyTorch model
    is_torch_model = TORCH_AVAILABLE and isinstance(autoencoder, nn.Module)
    
    if is_torch_model:
        autoencoder = autoencoder.to(device)
        autoencoder.eval()
    
    downscaled_frames = []
    
    with torch.no_grad() if is_torch_model else _nullcontext():
        for frame in frames:
            # Convert frame to tensor format expected by model
            # Frame is (H, W, 3) numpy array, convert to (1, 3, H, W) tensor
            frame_tensor = _frame_to_tensor(frame, device)
            
            try:
                # Try different autoencoder interfaces
                if is_torch_model:
                    # Direct call: model expects (B, C, H, W) and outputs (B, C, H', W')
                    output = autoencoder(frame_tensor)
                elif hasattr(autoencoder, 'encode') and hasattr(autoencoder, 'decode'):
                    # Encode-decode interface
                    encoded = autoencoder.encode(frame_tensor)
                    output = autoencoder.decode(encoded)
                elif hasattr(autoencoder, '__call__'):
                    # Callable interface
                    output = autoencoder(frame_tensor)
                else:
                    raise ValueError(
                        "Autoencoder must be a PyTorch nn.Module, have encode/decode methods, "
                        "or be callable"
                    )
                
                # Convert output back to numpy
                downscaled_frame = _tensor_to_frame(output, target_size)
                downscaled_frames.append(downscaled_frame)
                
            except Exception as e:
                logger.warning(
                    f"Autoencoder downscaling failed for frame, falling back to letterbox: {e}"
                )
                # Fallback to letterbox resize
                downscaled_frame = letterbox_resize(frame, target_size)
                downscaled_frames.append(downscaled_frame)
    
    return downscaled_frames


def _frame_to_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert numpy frame (H, W, 3) to PyTorch tensor (1, 3, H, W).
    
    Args:
        frame: Input frame as numpy array (H, W, 3), uint8 [0, 255] or float [0, 1]
        device: Target device (CPU or CUDA)
    
    Returns:
        Tensor of shape (1, 3, H, W), float32, normalized to [0, 1]
    """
    # Ensure frame is float32 and in [0, 1] range
    if frame.dtype == np.uint8:
        frame_float = frame.astype(np.float32) / 255.0
    else:
        frame_float = frame.astype(np.float32)
        if frame_float.max() > 1.0:
            frame_float = frame_float / 255.0
    
    # Convert (H, W, 3) to (3, H, W) then add batch dimension (1, 3, H, W)
    frame_tensor = torch.from_numpy(frame_float).permute(2, 0, 1).unsqueeze(0)
    return frame_tensor.to(device)


def _tensor_to_frame(tensor: torch.Tensor, target_size: int) -> np.ndarray:
    """
    Convert PyTorch tensor to numpy frame.
    
    Args:
        tensor: Input tensor, shape (B, C, H, W) or (C, H, W) or (H, W, C)
        target_size: Target size for output frame
    
    Returns:
        Numpy array (target_size, target_size, 3), uint8 [0, 255]
    """
    # Move to CPU and convert to numpy
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Handle different tensor shapes
    if tensor.dim() == 4:  # (B, C, H, W)
        tensor = tensor[0]  # Take first batch item
    if tensor.dim() == 3 and tensor.shape[0] == 3:  # (C, H, W)
        tensor = tensor.permute(1, 2, 0)  # Convert to (H, W, C)
    
    # Convert to numpy
    frame = tensor.numpy()
    
    # Ensure values are in [0, 1] range, then convert to [0, 255]
    frame = np.clip(frame, 0, 1)
    if frame.max() <= 1.0:
        frame = (frame * 255).astype(np.uint8)
    else:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    
    # Resize to target size if needed
    h, w = frame.shape[:2]
    if h != target_size or w != target_size:
        frame = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    return frame


class _nullcontext:
    """Context manager that does nothing (for non-torch case)."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False


def downscale_video_frames(
    frames: list,
    method: str = "letterbox",
    target_size: int = 224,
    autoencoder: Optional[object] = None
) -> list:
    """
    Downscale a list of video frames.
    
    Args:
        frames: List of frames (each is H, W, 3)
        method: Downscaling method ("letterbox" or "autoencoder")
        target_size: Target size for letterbox method
        autoencoder: Optional autoencoder model for autoencoder method
    
    Returns:
        List of downscaled frames
    """
    if method == "letterbox":
        return [letterbox_resize(frame, target_size) for frame in frames]
    elif method == "autoencoder":
        if autoencoder is None:
            raise ValueError("Autoencoder model required for autoencoder method")
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for autoencoder downscaling")
        
        return _autoencoder_downscale(frames, autoencoder, target_size)
    else:
        raise ValueError(f"Unknown downscaling method: {method}")

