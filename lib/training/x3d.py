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
        
        # Try to use torchvision's X3D if available
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
                logger.warning(f"X3D variant {variant} not available. Using x3d_m.")
                self.backbone = x3d_m(pretrained=pretrained)
            
            # Replace classification head for binary classification
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)
            self.use_torchvision = True
            
        except (ImportError, AttributeError):
            # Fallback: use r3d_18 as approximation
            logger.warning("torchvision X3D not available. Using r3d_18 as approximation.")
            from torchvision.models.video import r3d_18, R3D_18_Weights
            try:
                weights = R3D_18_Weights.KINETICS400_V1
                self.backbone = r3d_18(weights=weights)
            except (AttributeError, ValueError):
                self.backbone = r3d_18(pretrained=pretrained)
            
            # Replace classification head
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)
            self.use_torchvision = False
        
        self.variant = variant
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (N, C, T, H, W)
        
        Returns:
            Logits (N, 1)
        """
        return self.backbone(x)


__all__ = ["X3DModel"]

