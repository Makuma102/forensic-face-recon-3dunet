"""Dataset of preprocessed samples (one .npz per subject)."""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


def load_meta(path: str) -> dict:
    with np.load(path) as z:
        return json.loads(str(z["meta"]))


def coord_channels(shape) -> np.ndarray:
    grids = np.meshgrid(*[np.linspace(-1, 1, n, dtype=np.float32) for n in shape], indexing="ij")
    return np.stack(grids)


class SkullFaceDataset(Dataset):
    def __init__(self, files: list[str], trunc_mm: float = 20.0, coords: bool = False,
                 augment: bool = False, input_key: str = "skull_sdf", seed: int = 0,
                 patch: tuple | None = None, surface_bias: float = 0.8):
        self.files = files
        self.T = trunc_mm
        self.coords = coords
        self.augment = augment
        self.input_key = input_key
        self.rng = np.random.default_rng(seed)
        self.patch = tuple(patch) if patch else None
        self.surface_bias = surface_bias

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with np.load(self.files[i]) as z:
            x = z[self.input_key].astype(np.float32) / self.T
            y = z["skin_sdf"].astype(np.float32) / self.T
            valid = z["valid"].astype(np.float32)
        if self.augment:
            # Left-right mirroring is anatomically plausible for a face and keeps
            # the crop frame; shifts of a few voxels mimic anchor error.
            if self.rng.random() < 0.5:
                x, y, valid = x[..., ::-1], y[..., ::-1], valid[..., ::-1]
            sh = self.rng.integers(-2, 3, size=3)
            x = np.roll(x, sh, (0, 1, 2))
            y = np.roll(y, sh, (0, 1, 2))
            valid = np.roll(valid, sh, (0, 1, 2))
        inp = x[None]
        if self.coords:
            # Coordinates of the whole crop, so a patch still knows where it sits.
            inp = np.concatenate([inp, coord_channels(x.shape)], 0)
        y, valid = y[None], valid[None]
        if self.patch:
            sl = self._patch_slices(y[0])
            inp, y, valid = inp[(slice(None), *sl)], y[(slice(None), *sl)], valid[(slice(None), *sl)]
        return (torch.from_numpy(np.ascontiguousarray(inp)),
                torch.from_numpy(np.ascontiguousarray(y)),
                torch.from_numpy(np.ascontiguousarray(valid)),
                os.path.basename(self.files[i]).replace(".npz", ""))

    def _patch_slices(self, y: np.ndarray):
        """Random patch, centred near the true skin surface with probability ``surface_bias``."""
        shape = np.array(y.shape)
        ps = np.minimum(np.array(self.patch), shape)
        if self.rng.random() < self.surface_bias:
            cand = np.argwhere(np.abs(y) < 3.0 / self.T)
            c = cand[self.rng.integers(len(cand))] if len(cand) else shape // 2
        else:
            c = np.array([self.rng.integers(n) for n in shape])
        lo = np.clip(c - ps // 2, 0, shape - ps)
        return tuple(slice(int(a), int(a + b)) for a, b in zip(lo, ps))
