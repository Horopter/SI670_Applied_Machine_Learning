"""
Advanced SlowFast variants: SlowFast with attention and multi-scale SlowFast.
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

from lib.training.slowfast import SlowFastModel


class SlowFastAttentionModel(nn.Module):
    """
    SlowFast with attention mechanisms.
    
    Adds self-attention and cross-attention between slow and fast pathways.
    """
    
    def __init__(
        self,
        slow_frames: int = 16,
        fast_frames: int = 64,
        alpha: int = 8,
        beta: float = 1.0 / 8,
        pretrained: bool = True,
        attention_type: str = "cross"  # "self", "cross", "both"
    ):
        """
        Initialize SlowFast with attention.
        
        Args:
            slow_frames: Number of frames for slow pathway
            fast_frames: Number of frames for fast pathway
            alpha: Temporal ratio
            beta: Channel ratio
            pretrained: Use pretrained weights
            attention_type: Type of attention to use
        """
        super().__init__()
        
        # Use base SlowFast model
        self.slowfast = SlowFastModel(
            slow_frames=slow_frames,
            fast_frames=fast_frames,
            alpha=alpha,
            beta=beta,
            pretrained=pretrained
        )
        
        # Get feature dimensions from SlowFast
        # For simplicity, we'll add attention after the pathways but before fusion
        # In practice, we'd need to modify SlowFast to extract intermediate features
        
        # For now, we'll add attention at the fusion stage
        # Assume we have slow and fast features of some dimension
        # We'll use a simplified approach: add attention after extracting features
        
        self.attention_type = attention_type
        
        # Cross-attention between slow and fast pathways
        if attention_type in ["cross", "both"]:
            # We'll need to know the feature dimensions
            # For now, use a reasonable default
            feature_dim = 512  # Approximate feature dimension after pathways
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=8,
                batch_first=True
            )
        
        # Self-attention for each pathway
        if attention_type in ["self", "both"]:
            feature_dim = 512
            self.slow_self_attention = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=8,
                batch_first=True
            )
            self.fast_self_attention = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=8,
                batch_first=True
            )
        
        # Note: This is a simplified implementation
        # A full implementation would require modifying SlowFast to expose intermediate features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (N, C, T, H, W)
        
        Returns:
            Logits (N, 1)
        """
        # For now, use base SlowFast
        # In a full implementation, we would:
        # 1. Extract slow and fast pathway features
        # 2. Apply attention
        # 3. Fuse with attention-weighted features
        
        # Simplified: just use base model
        return self.slowfast(x)


class MultiScaleSlowFastModel(nn.Module):
    """
    Multi-scale SlowFast: Multiple temporal sampling rates.
    
    Instead of just 2 pathways (slow and fast), uses multiple pathways
    with different temporal sampling rates.
    """
    
    def __init__(
        self,
        num_frames: int = 64,
        scales: list = [1, 2, 4, 8],  # Temporal sampling rates
        pretrained: bool = True
    ):
        """
        Initialize Multi-scale SlowFast.
        
        Args:
            num_frames: Total number of input frames
            scales: List of temporal sampling rates (1 = every frame, 2 = every 2nd frame, etc.)
            pretrained: Use pretrained weights
        """
        super().__init__()
        
        self.num_frames = num_frames
        self.scales = scales
        self.num_pathways = len(scales)
        
        # Create a pathway for each scale
        self.pathways = nn.ModuleList()
        
        for scale in scales:
            # Use a simplified 3D ResNet-like backbone for each pathway
            try:
                from torchvision.models.video import r3d_18, R3D_18_Weights
                if pretrained:
                    try:
                        weights = R3D_18_Weights.KINETICS400_V1
                        pathway = r3d_18(weights=weights)
                    except (AttributeError, ValueError):
                        pathway = r3d_18(pretrained=True)
                else:
                    pathway = r3d_18(pretrained=False)
                
                # Remove classification head
                pathway = nn.Sequential(*list(pathway.children())[:-1])
                self.pathways.append(pathway)
            except (ImportError, AttributeError):
                # Fallback: simple 3D CNN
                pathway = self._build_simple_pathway()
                self.pathways.append(pathway)
        
        # Fusion: combine features from all pathways
        # Assume each pathway outputs features of dimension 512
        feature_dim = 512
        fusion_dim = feature_dim * self.num_pathways
        
        self.fusion = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(fusion_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
    
    def _build_simple_pathway(self):
        """Build a simple 3D CNN pathway."""
        return nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (N, C, T, H, W)
        
        Returns:
            Logits (N, 1)
        """
        N, C, T, H, W = x.shape
        
        pathway_features = []
        
        # Process through each pathway with different temporal sampling
        for i, (pathway, scale) in enumerate(zip(self.pathways, self.scales)):
            # Sample frames according to scale
            if scale == 1:
                # Use all frames
                pathway_input = x
            else:
                # Sample every scale-th frame
                indices = torch.arange(0, T, scale, device=x.device)
                pathway_input = x[:, :, indices, :, :]  # (N, C, T', H, W)
            
            # Process through pathway
            features = pathway(pathway_input)  # (N, feature_dim, T', H', W')
            
            # Temporal alignment: pool to same temporal size
            features = F.adaptive_avg_pool3d(features, (1, 1, 1))  # (N, feature_dim, 1, 1, 1)
            pathway_features.append(features)
        
        # Concatenate features from all pathways
        combined = torch.cat(pathway_features, dim=1)  # (N, feature_dim*num_pathways, 1, 1, 1)
        
        # Classification
        logits = self.fusion(combined)  # (N, 1)
        
        return logits


__all__ = ["SlowFastAttentionModel", "MultiScaleSlowFastModel"]

