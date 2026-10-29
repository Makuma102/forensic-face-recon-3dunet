"""Frontal renderings of a predicted face for investigators: shaded view and a sketch-style view.

Works directly on an SDF grid (z, y, x; LPS crop, so the face looks toward -y).
The sketch view is meant as a neutral starting point for a forensic artist's
composite, not a photorealistic likeness: hair, skin colour, eyes and
expression are not predicted from bone and are deliberately left out.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def frontal_depth(sdf: np.ndarray, spacing: float, smooth_px: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    """Depth (mm from the crop front) of the first skin surface along +y, with sub-voxel interpolation."""
    inside = sdf < 0
    hit = inside.any(1)                          # (z, x)
    first = np.argmax(inside, axis=1)            # first inside voxel along y
    z, x = np.nonzero(hit)
    j = first[z, x]
    prev = np.clip(j - 1, 0, None)
    a, b = sdf[z, prev, x], sdf[z, j, x]         # a >= 0 > b
    t = np.where(a - b > 1e-6, a / (a - b + 1e-9), 0.0)
    depth = np.full(hit.shape, np.nan, np.float32)
    depth[z, x] = (prev + t) * spacing
    # Voxelised surfaces show terracing; a light normalised blur removes it.
    w = ndimage.gaussian_filter(hit.astype(np.float32), smooth_px)
    d = ndimage.gaussian_filter(np.nan_to_num(depth) * hit, smooth_px)
    depth[hit] = (d / np.maximum(w, 1e-6))[hit]
    return depth, hit


def shaded(depth: np.ndarray, spacing: float, light=(0.4, -0.5, 0.75)) -> np.ndarray:
    """Lambertian shading of the depth map; returns values in [0, 1], image rows = superior at top."""
    d = np.nan_to_num(depth, nan=np.nanmax(depth))
    gz, gx = np.gradient(d, spacing)
    n = np.stack([-gz, -gx, np.ones_like(d)], -1)       # (up, right, toward viewer)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    lz, lx, lv = light
    L = np.array([lz, lx, lv]) / np.linalg.norm(light)
    img = np.clip(n @ L, 0, 1)
    img[np.isnan(depth)] = 1.0
    return img[::-1]


def sketch(depth: np.ndarray, spacing: float) -> np.ndarray:
    """Pencil-style rendering: soft tone from shading plus lines at creases and silhouettes."""
    tone = shaded(depth, spacing)
    d = np.nan_to_num(depth, nan=np.nanmax(depth))[::-1]
    # Lines where depth bends sharply (crease/contour lines) and at the silhouette.
    lap = np.abs(ndimage.gaussian_laplace(d, sigma=1.2))
    lines = np.clip(lap / (np.percentile(lap, 97) + 1e-6), 0, 1)
    sil = ndimage.binary_dilation(np.isnan(depth[::-1])) ^ np.isnan(depth[::-1])
    tone = 1 - 0.45 * (1 - tone)                      # lighten the shading like graphite
    out = tone * (1 - 0.8 * lines)
    out[sil] = 0.15
    out[np.isnan(depth[::-1])] = 1.0
    return np.clip(out, 0, 1)


def to_uint8(img: np.ndarray, upscale: int = 3) -> np.ndarray:
    im = (img * 255).astype(np.uint8)
    return np.kron(im, np.ones((upscale, upscale), np.uint8)) if upscale > 1 else im
