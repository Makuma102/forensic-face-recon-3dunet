"""Losses on truncated, normalised signed-distance fields (values in [-1, 1])."""
from __future__ import annotations

import torch


def sdf_loss(pred, target, valid=None, surface_weight: float = 4.0, sigma: float = 0.15,
             sign_weight: float = 0.0):
    """
    L1 on the SDF, up-weighted near the target surface where the face actually is.

    surface_weight: extra weight w*exp(-|t|/sigma) on voxels close to the skin.
    sign_weight:    optional hinge on wrong-side predictions (inside vs outside).
    valid:          mask of voxels inside the CT field of view; others are ignored.
    """
    w = 1.0 + surface_weight * torch.exp(-target.abs() / sigma)
    if valid is not None:
        w = w * valid
    l1 = (w * (pred - target).abs()).sum() / w.sum().clamp_min(1.0)
    if sign_weight > 0:
        hinge = torch.relu(-pred * torch.sign(target))
        l1 = l1 + sign_weight * (w * hinge).sum() / w.sum().clamp_min(1.0)
    return l1
