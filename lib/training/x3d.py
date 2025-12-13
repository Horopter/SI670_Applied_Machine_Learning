"""
X3D model: Expanding Architectures for Efficient Video Recognition.
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class X3DModel(nn.Module):
    """
    X3D (Expanding Architectures for Efficient Video Recognition) model.
    """
    
    def __init__(
        self,
        variant: str = "x3d_m",  # "x3d_s", "x3d_m", "x3d_l", "x3d_xl"
        pretrained: bool = True
    ):
        """
        Initialize X3D model.
        
        Args:
            variant: X3D variant (x3d_s, x3d_m, x3d_l, x3d_xl)
            pretrained: Use pretrained weights
        """
        super().__init__()
        
        # Try PyTorchVideo first (recommended method for X3D)
        backbone_loaded = False
        try:
            # Map variant names to PyTorchVideo model names
            pytorchvideo_model_map = {
                "x3d_xs": "x3d_xs",
                "x3d_s": "x3d_s",
                "x3d_m": "x3d_m",
                "x3d_l": "x3d_l",
            }
            
            pytorchvideo_model_name = pytorchvideo_model_map.get(variant, "x3d_m")
            
            if pretrained:
                logger.info(f"Loading X3D model from PyTorchVideo: {pytorchvideo_model_name} (pretrained=True)")
                self.backbone = torch.hub.load(
                    'facebookresearch/pytorchvideo',
                    pytorchvideo_model_name,
                    pretrained=True
                )
            else:
                logger.info(f"Loading X3D model from PyTorchVideo: {pytorchvideo_model_name} (pretrained=False)")
                self.backbone = torch.hub.load(
                    'facebookresearch/pytorchvideo',
                    pytorchvideo_model_name,
                    pretrained=False
                )
            
            # Replace classification head for binary classification
            # PyTorchVideo X3D models typically have a head with a projection layer
            # Try multiple strategies to find and replace the final classification layer
            head_replaced = False
            
            # Strategy 1: Direct head.proj access (most common for PyTorchVideo X3D)
            if hasattr(self.backbone, 'head'):
                if hasattr(self.backbone.head, 'proj') and isinstance(self.backbone.head.proj, nn.Linear):
                    in_features = self.backbone.head.proj.in_features
                    self.backbone.head.proj = nn.Linear(in_features, 1)
                    head_replaced = True
                elif isinstance(self.backbone.head, nn.Linear):
                    # Head is directly a Linear layer
                    in_features = self.backbone.head.in_features
                    self.backbone.head = nn.Linear(in_features, 1)
                    head_replaced = True
            
            # Strategy 2: Find the last Linear layer in the model
            if not head_replaced:
                last_linear_name = None
                last_linear_in_features = None
                # First pass: find the last Linear layer
                for name, module in self.backbone.named_modules():
                    if isinstance(module, nn.Linear):
                        last_linear_name = name
                        last_linear_in_features = module.in_features
                
                # Second pass: replace it
                if last_linear_name is not None:
                    parts = last_linear_name.split('.')
                    if len(parts) > 1:
                        parent = self.backbone
                        for part in parts[:-1]:
                            parent = getattr(parent, part)
                        setattr(parent, parts[-1], nn.Linear(last_linear_in_features, 1))
                    else:
                        setattr(self.backbone, last_linear_name, nn.Linear(last_linear_in_features, 1))
                    head_replaced = True
                    logger.debug(f"Replaced classification head at: {last_linear_name}")
            
            if not head_replaced:
                logger.warning("Could not find classification head in PyTorchVideo X3D model. Model may not work correctly.")
            
            self.use_pytorchvideo = True
            backbone_loaded = True
            logger.info(f"✓ Successfully loaded X3D model from PyTorchVideo: {pytorchvideo_model_name}")
            
        except Exception as e:
            logger.warning(f"Failed to load X3D from PyTorchVideo: {e}. Trying torchvision...")
            backbone_loaded = False
        
        # Fallback to torchvision X3D if PyTorchVideo failed
        if not backbone_loaded:
            try:
                from torchvision.models.video import x3d_m, X3D_M_Weights
                
                if variant == "x3d_m":
                    if pretrained:
                        try:
                            weights = X3D_M_Weights.KINETICS400_V1
                            self.backbone = x3d_m(weights=weights)
                        except (AttributeError, ValueError):
                            self.backbone = x3d_m(pretrained=True)
                    else:
                        self.backbone = x3d_m(pretrained=False)
                else:
                    # For other variants, try to load or use x3d_m as fallback
                    logger.warning(f"X3D variant {variant} not available in torchvision. Using x3d_m.")
                    self.backbone = x3d_m(pretrained=pretrained)
                
                # Replace classification head for binary classification
                self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)
                self.use_pytorchvideo = False
                self.use_torchvision = True
                backbone_loaded = True
                logger.info("✓ Successfully loaded X3D model from torchvision")
                
            except (ImportError, AttributeError) as e:
                logger.warning(f"torchvision X3D not available: {e}. Using r3d_18 as approximation.")
                backbone_loaded = False
        
        # Final fallback: use r3d_18 as approximation
        if not backbone_loaded:
            try:
                from torchvision.models.video import r3d_18, R3D_18_Weights
                try:
                    weights = R3D_18_Weights.KINETICS400_V1
                    self.backbone = r3d_18(weights=weights)
                except (AttributeError, ValueError):
                    self.backbone = r3d_18(pretrained=pretrained)
                
                # Replace classification head
                self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)
                self.use_pytorchvideo = False
                self.use_torchvision = False
                logger.warning("⚠ Using r3d_18 as X3D approximation (X3D not available)")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load X3D model or fallback. "
                    f"Tried PyTorchVideo, torchvision, and r3d_18. Last error: {e}. "
                    f"Please install pytorchvideo: pip install pytorchvideo"
                ) from e
        
        self.variant = variant
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (N, C, T, H, W)
        
        Returns:
            Logits (N, 1)
        """
        # CRITICAL: X3D requires minimum spatial dimensions (32x32) for pooling kernels
        # The error "input image (T: 500 H: 5 W: 8) smaller than kernel size (kT: 16 kH: 7 kW: 7)"
        # indicates some videos have extremely small spatial dimensions after scaling
        # We MUST resize these BEFORE passing to backbone to prevent crashes
        # 
        # NOTE: We are UPSCALING small inputs (e.g., 5x8 -> 32x51), NOT downscaling large inputs
        # This INCREASES memory usage per sample, so batch size should remain conservative
        if x.dim() == 5:
            N, C, T, H, W = x.shape
            min_spatial_size = 32  # Minimum required for X3D (kernel size is 7x7, but needs buffer for pooling)
            
            # CRITICAL: Always resize if spatial dimensions are too small
            # This handles cases where videos have H < 32 or W < 32 (e.g., H=5, W=8)
            # We UPSCALE these to meet minimum requirements (increases memory usage)
            if H < min_spatial_size or W < min_spatial_size:
                import torch.nn.functional as F
                
                # Calculate target size maintaining aspect ratio
                # For very small inputs, we need to scale up significantly
                if H <= 0 or W <= 0:
                    # Invalid dimensions - use default minimum size
                    new_h = min_spatial_size
                    new_w = min_spatial_size
                else:
                    # Maintain aspect ratio while ensuring minimum size
                    # Scale the smaller dimension to min_spatial_size, then scale the larger proportionally
                    if H < W:
                        # Height is smaller - scale height to min_spatial_size
                        scale_factor = min_spatial_size / max(H, 1.0)
                        new_h = min_spatial_size
                        new_w = max(min_spatial_size, int(W * scale_factor))
                    else:
                        # Width is smaller - scale width to min_spatial_size
                        scale_factor = min_spatial_size / max(W, 1.0)
                        new_w = min_spatial_size
                        new_h = max(min_spatial_size, int(H * scale_factor))
                
                # Safety check: ensure both dimensions are at least min_spatial_size
                new_h = max(new_h, min_spatial_size)
                new_w = max(new_w, min_spatial_size)
                
                # Resize: (N, C, T, H, W) -> (N*T, C, H, W) -> resize -> (N, C, T, H', W')
                # This preserves temporal dimension while resizing spatial dimensions
                x_reshaped = x.permute(0, 2, 1, 3, 4).contiguous()  # (N, T, C, H, W)
                x_reshaped = x_reshaped.view(N * T, C, H, W)  # (N*T, C, H, W)
                
                # Use bilinear interpolation for smooth resizing
                # This handles even very small inputs (e.g., 5x8 -> 32x51)
                x_resized = F.interpolate(
                    x_reshaped, 
                    size=(new_h, new_w), 
                    mode='bilinear', 
                    align_corners=False
                )  # (N*T, C, H', W')
                
                # Reshape back to (N, C, T, H', W')
                x_resized = x_resized.view(N, T, C, new_h, new_w)  # (N, T, C, H', W')
                x = x_resized.permute(0, 2, 1, 3, 4).contiguous()  # (N, C, T, H', W')
                
                # Log resize operation for debugging (only for very small inputs to avoid spam)
                if H < 16 or W < 16:
                    logger.debug(
                        f"X3D: Resized input from {H}x{W} to {new_h}x{new_w} "
                        f"(temporal: {T} frames) to meet minimum spatial dimension requirements"
                    )
        
        return self.backbone(x)


__all__ = ["X3DModel"]

