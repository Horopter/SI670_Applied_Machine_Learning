"""
Simple Grad-CAM-style explanation utilities for the FVC video classifier.

Generates a coarse spatio-temporal heatmap indicating which regions the model
attended to for its prediction.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from .video_modeling import VariableARVideoModel


@torch.no_grad()
def grad_cam_3d(
    model: VariableARVideoModel,
    clips: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute a simple Grad-CAM heatmap for VariableARVideoModel on given clips.

    Args:
        model: trained VariableARVideoModel
        clips: (N, C, T, H, W)

    Returns:
        heatmaps: (N, 1, T, H, W) normalized to [0, 1]
        preds: (N,) predicted labels
    """
    model.eval()
    device = next(model.parameters()).device
    clips = clips.to(device)

    # We will hook the output of the last Inception block.
    activations = []
    gradients = []

    def fwd_hook(_, __, output):
        activations.append(output)

    def bwd_hook(_, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle_fwd = model.incept2.register_forward_hook(fwd_hook)
    handle_bwd = model.incept2.register_backward_hook(bwd_hook)

    clips.requires_grad_(True)
    logits = model(clips).squeeze(-1)
    probs = logits.sigmoid()
    preds = (probs >= 0.5).long()

    # For Grad-CAM, backprop the max logit per sample
    target = logits
    model.zero_grad()
    target.sum().backward()

    handle_fwd.remove()
    handle_bwd.remove()

    act = activations[0]        # (N, C, T, H, W)
    grad = gradients[0]         # (N, C, T, H, W)

    # Global average pool gradients over T,H,W to obtain channel weights
    weights = grad.mean(dim=(2, 3, 4), keepdim=True)  # (N, C, 1, 1, 1)
    cam = (weights * act).sum(dim=1, keepdim=True)    # (N, 1, T, H, W)
    cam = F.relu(cam)

    # Normalize each heatmap to [0, 1]
    N = cam.shape[0]
    heatmaps = []
    for i in range(N):
        h = cam[i]
        h_min, h_max = h.min(), h.max()
        if (h_max - h_min) > 0:
            h = (h - h_min) / (h_max - h_min)
        heatmaps.append(h)
    heatmaps = torch.stack(heatmaps, dim=0)

    return heatmaps.detach().cpu(), preds.detach().cpu()


__all__ = ["grad_cam_3d"]


