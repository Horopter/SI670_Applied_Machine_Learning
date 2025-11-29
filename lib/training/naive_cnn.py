"""
Naive CNN baseline model that processes frames independently.
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class NaiveCNNBaseline(nn.Module):
    """
    Naive CNN baseline that processes frames independently and averages predictions.
    """
    
    def __init__(self, num_frames: int = 8, num_classes: int = 2):
        """
        Initialize CNN model.
        
        Args:
            num_frames: Number of frames to process
            num_classes: Number of classes (2 for binary)
        """
        super().__init__()
        
        # Simple 2D CNN for per-frame processing
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
        
        self.num_frames = num_frames
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (N, C, T, H, W) or (N, T, C, H, W)
        
        Returns:
            Logits (N, num_classes)
        """
        # Handle different input formats
        if x.dim() == 5:
            if x.shape[1] == 3:  # (N, C, T, H, W)
                # Rearrange to (N*T, C, H, W) for per-frame processing
                N, C, T, H, W = x.shape
                x = x.permute(0, 2, 1, 3, 4).contiguous()  # (N, T, C, H, W)
                x = x.view(N * T, C, H, W)
            else:  # (N, T, C, H, W)
                N, T, C, H, W = x.shape
                x = x.view(N * T, C, H, W)
        else:
            # Already in (N*T, C, H, W) format
            pass
        
        # Process each frame independently
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)  # (N*T, num_classes)
        
        # Reshape back to (N, T, num_classes) and average over frames
        if x.dim() == 2 and logits.shape[0] % self.num_frames == 0:
            N = logits.shape[0] // self.num_frames
            logits = logits.view(N, self.num_frames, -1)
            logits = logits.mean(dim=1)  # Average over frames
        
        return logits


__all__ = ["NaiveCNNBaseline"]

