"""Non-deep baselines that a reviewer will expect next to the U-Net.

All operate in the same skull-anchored crop and return a skin SDF in mm on the
network grid, so they are evaluated by exactly the same code as the model.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .preprocess import _ball, signed_distance


def _load(files, key):
    out = []
    for f in files:
        with np.load(f) as z:
            out.append(z[key].astype(np.float32))
    return np.stack(out)


class MeanFace:
    """Population-average face in the skull-anchored frame (ignores the skull's shape)."""
    name = "mean_face"

    def fit(self, train_files):
        self.mean = _load(train_files, "skin_sdf").mean(0)
        return self

    def predict(self, skull_sdf, spacing):
        return self.mean


def skull_envelope(skull_sdf: np.ndarray, spacing: float, close_mm: float = 6.0) -> np.ndarray:
    """Solid region bounded by the outer bone surface (cranial and facial cavities filled)."""
    bone = skull_sdf < 0
    r = max(1, int(round(close_mm / spacing)))
    env = ndimage.binary_closing(np.pad(bone, r), structure=_ball(r))[r:-r, r:-r, r:-r]
    env = ndimage.binary_fill_holes(env)
    for z in range(env.shape[0]):
        env[z] = ndimage.binary_fill_holes(env[z])
    return env


class ConstantTissueDepth:
    """Classical tissue-depth idea in its simplest form: offset the skull envelope by one global depth.

    The depth is chosen on the training set to minimise mean |SDF error| near the true skin.
    """
    name = "tissue_depth"

    def __init__(self, depths_mm=np.arange(2.0, 14.1, 0.5)):
        self.depths = depths_mm

    def fit(self, train_files, spacing=2.0):
        errs = np.zeros(len(self.depths))
        for f in train_files:
            with np.load(f) as z:
                s, t, v = z["skull_sdf"].astype(np.float32), z["skin_sdf"].astype(np.float32), z["valid"]
            env = signed_distance(skull_envelope(s, spacing), spacing)
            near = (np.abs(t) < 4) & v
            for i, d in enumerate(self.depths):
                errs[i] += np.abs(env[near] - d - t[near]).mean()
        self.depth = float(self.depths[np.argmin(errs)])
        return self

    def predict(self, skull_sdf, spacing):
        return signed_distance(skull_envelope(skull_sdf, spacing), spacing) - self.depth


class NearestNeighbour:
    """Return the training face whose skull is most similar (L1 between skull SDFs)."""
    name = "nearest_neighbour"

    def fit(self, train_files):
        self.X = _load(train_files, "skull_sdf")
        self.Y = _load(train_files, "skin_sdf")
        return self

    def predict(self, skull_sdf, spacing):
        d = np.abs(self.X - skull_sdf[None]).mean((1, 2, 3))
        return self.Y[int(np.argmin(d))]


class SkullFaceRidge:
    """Linear statistical model (PCA/ridge family, cf. Claes et al.): skin = mean + linear map of skull.

    Solved in dual form, so it is exact for small N with millions of voxels.
    """
    name = "ridge"

    def __init__(self, alpha_grid=(1e-1, 1, 10, 100, 1000)):
        self.alpha_grid = alpha_grid

    def fit(self, train_files):
        X = _load(train_files, "skull_sdf").reshape(len(train_files), -1)
        Y = _load(train_files, "skin_sdf").reshape(len(train_files), -1)
        self.xm, self.ym = X.mean(0), Y.mean(0)
        self.Xc, Yc = X - self.xm, Y - self.ym
        K = self.Xc @ self.Xc.T / X.shape[1]
        n = len(K)
        # Leave-one-out choice of alpha (closed form for kernel ridge).
        best = None
        for a in self.alpha_grid:
            H = K @ np.linalg.inv(K + a * np.eye(n))
            loo = ((Yc - H @ Yc) / (1 - np.diag(H))[:, None])
            e = np.abs(loo).mean()
            if best is None or e < best[0]:
                best = (e, a)
        self.alpha = best[1]
        self.A = np.linalg.solve(K + self.alpha * np.eye(n), Yc)
        self.shape = _load(train_files[:1], "skull_sdf").shape[1:]
        return self

    def predict(self, skull_sdf, spacing):
        k = self.Xc @ (skull_sdf.reshape(-1) - self.xm) / self.Xc.shape[1]
        return (self.ym + k @ self.A).reshape(self.shape)


BASELINES = {c.name: c for c in (MeanFace, ConstantTissueDepth, NearestNeighbour, SkullFaceRidge)}
