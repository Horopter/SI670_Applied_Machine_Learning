"""
R(2+1)D model: Factorized 3D convolutions for video classification.
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class R2Plus1DModel(nn.Module):
    """
    R(2+1)D (Factorized 3D Convolutions) model for video classification.
    """
    
    def __init__(self, pretrained: bool = True):
        """
        Initialize R(2+1)D model.
        
        Args:
            pretrained: Use pretrained weights from Kinetics-400
        """
        super().__init__()
        
        try:
            from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
            
            if pretrained:
                try:
                    weights = R2Plus1D_18_Weights.KINETICS400_V1
                    self.backbone = r2plus1d_18(weights=weights)
                except (AttributeError, ValueError):
                    self.backbone = r2plus1d_18(pretrained=True)
            else:
                self.backbone = r2plus1d_18(pretrained=False)
            
            # Replace classification head for binary classification
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)
            self.use_torchvision = True
            
        except ImportError:
            logger.warning("torchvision R(2+1)D not available. Using fallback implementation.")
            # Fallback: use R3D_18 as approximation
            from torchvision.models.video import r3d_18, R3D_18_Weights
            if pretrained:
                try:
                    weights = R3D_18_Weights.KINETICS400_V1
                    self.backbone = r3d_18(weights=weights)
                except (AttributeError, ValueError):
                    self.backbone = r3d_18(pretrained=True)
            else:
                self.backbone = r3d_18(pretrained=False)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)
            self.use_torchvision = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (N, C, T, H, W)
        
        Returns:
            Logits (N, 1)
        """
        # CRITICAL: Handle small spatial dimensions like X3D, SlowFast, and I3D
        # R(2+1)D also requires minimum spatial dimensions for proper processing
        if x.dim() == 5:
            N, C, T, H, W = x.shape
            min_spatial_size = 32  # Minimum required for R(2+1)D
            
            if H < min_spatial_size or W < min_spatial_size:
                import torch.nn.functional as F
                
                if H <= 0 or W <= 0:
                    new_h = min_spatial_size
                    new_w = min_spatial_size
                else:
                    if H < W:
                        scale_factor = min_spatial_size / max(H, 1.0)
                        new_h = min_spatial_size
                        new_w = max(min_spatial_size, int(W * scale_factor))
                    else:
                        scale_factor = min_spatial_size / max(W, 1.0)
                        new_w = min_spatial_size
                        new_h = max(min_spatial_size, int(H * scale_factor))
                
                new_h = max(new_h, min_spatial_size)
                new_w = max(new_w, min_spatial_size)
                
                x_reshaped = x.permute(0, 2, 1, 3, 4).contiguous()  # (N, T, C, H, W)
                x_reshaped = x_reshaped.view(N * T, C, H, W)  # (N*T, C, H, W)
                x_resized = F.interpolate(
                    x_reshaped, 
                    size=(new_h, new_w), 
                    mode='bilinear', 
                    align_corners=False
                )  # (N*T, C, H', W')
                x_resized = x_resized.view(N, T, C, new_h, new_w)  # (N, T, C, H', W')
                x = x_resized.permute(0, 2, 1, 3, 4).contiguous()  # (N, C, T, H', W')
        
        return self.backbone(x)


__all__ = ["R2Plus1DModel"]

