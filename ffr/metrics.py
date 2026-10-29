"""Surface-based evaluation in millimetres and a shape-based identification test."""
from __future__ import annotations

import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import measure


def sdf_to_mesh(sdf: np.ndarray, spacing: float, level: float = 0.0) -> trimesh.Trimesh | None:
    """Iso-surface of an SDF (array z,y,x) as a mesh with vertices in mm, (z, y, x) order."""
    if not (sdf.min() < level < sdf.max()):
        return None
    v, f, _, _ = measure.marching_cubes(sdf.astype(np.float32), level=level, spacing=(spacing,) * 3)
    m = trimesh.Trimesh(v, f, process=False)
    # Keep the largest connected piece (the head); drop floating specks.
    parts = m.split(only_watertight=False)
    return max(parts, key=lambda p: len(p.faces)) if len(parts) > 1 else m


def upsample_to(sdf: np.ndarray, shape) -> np.ndarray:
    f = [t / s for t, s in zip(shape, sdf.shape)]
    out = ndimage.zoom(sdf.astype(np.float32), f, order=1)
    return out[: shape[0], : shape[1], : shape[2]]


def face_roi(points_mm: np.ndarray, roi: dict) -> np.ndarray:
    """Boolean mask of points in the anterior-face region (crop coordinates, mm).

    roi keys: y_max_mm (depth behind the crop's front face), z_min_mm, z_max_mm
    (heights above the crop's bottom), border_mm (ignore crop borders),
    size_mm (crop size z, y, x).
    """
    z, y, x = points_mm.T
    b = roi.get("border_mm", 3.0)
    sz, sy, sx = roi["size_mm"]
    m = (y <= roi["y_max_mm"]) & (z >= roi["z_min_mm"]) & (z <= roi["z_max_mm"])
    m &= (z > b) & (z < sz - b) & (y > b) & (y < sy - b) & (x > b) & (x < sx - b)
    return m


def surface_metrics(pred: trimesh.Trimesh, gt: trimesh.Trimesh, roi: dict, gt_sdf_fine: np.ndarray | None = None,
                    spacing_fine: float = 1.0, valid_fine: np.ndarray | None = None,
                    n_samples: int = 200_000, seed: int = 0) -> dict:
    """Symmetric surface distances restricted to the face ROI (mm)."""
    rng = np.random.default_rng(seed)
    p_pts, _ = trimesh.sample.sample_surface(pred, n_samples, seed=rng.integers(1 << 31))
    g_pts, _ = trimesh.sample.sample_surface(gt, n_samples, seed=rng.integers(1 << 31))
    keep_p, keep_g = face_roi(p_pts, roi), face_roi(g_pts, roi)
    if valid_fine is not None:
        def inside(pts):
            idx = np.clip(np.round(pts / spacing_fine).astype(int), 0, np.array(valid_fine.shape) - 1)
            return valid_fine[idx[:, 0], idx[:, 1], idx[:, 2]]
        keep_p &= inside(p_pts)
        keep_g &= inside(g_pts)
    p_roi, g_roi = p_pts[keep_p], g_pts[keep_g]
    if len(p_roi) == 0 or len(g_roi) == 0:
        return dict(n_pred=len(p_roi), n_gt=len(g_roi))
    d_pg = cKDTree(g_pts).query(p_roi)[0]   # pred -> GT (GT surface sampled densely, ~0.3 mm)
    d_gp = cKDTree(p_pts).query(g_roi)[0]   # GT -> pred
    both = np.concatenate([d_pg, d_gp])
    out = dict(
        mean_mm=float(both.mean()), median_mm=float(np.median(both)), rms_mm=float(np.sqrt((both ** 2).mean())),
        p95_mm=float(np.percentile(both, 95)), hd95_mm=float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95))),
        within_2mm=float((both <= 2).mean()), within_5mm=float((both <= 5).mean()),
        pred_to_gt_mean_mm=float(d_pg.mean()), gt_to_pred_mean_mm=float(d_gp.mean()),
        n_pred=int(len(p_roi)), n_gt=int(len(g_roi)),
    )
    if gt_sdf_fine is not None:
        # Signed error at predicted surface: >0 means predicted skin lies outside the true skin.
        idx = p_roi / spacing_fine
        s = ndimage.map_coordinates(gt_sdf_fine.astype(np.float32), idx.T, order=1, mode="nearest")
        out["signed_mean_mm"] = float(s.mean())
    # Error by facial third (crude height bands inside the ROI; landmarks are better).
    z0, z1 = roi["z_min_mm"], roi["z_max_mm"]
    edges = np.linspace(z0, z1, 4)
    for name, lo, hi in zip(["lower", "middle", "upper"], edges[:-1], edges[1:]):
        sel = (g_roi[:, 0] >= lo) & (g_roi[:, 0] < hi)
        out[f"{name}_third_mm"] = float(d_gp[sel].mean()) if sel.any() else float("nan")
    return out


def _roi_points(mesh: trimesh.Trimesh, roi: dict, n: int, seed: int) -> np.ndarray:
    pts, _ = trimesh.sample.sample_surface(mesh, n * 4, seed=seed)
    pts = pts[face_roi(pts, roi)]
    return pts[np.random.default_rng(seed).permutation(len(pts))[:n]]


def identification(preds: dict, gts: dict, roi: dict, n_points: int = 4000, icp: bool = True) -> dict:
    """
    Closed-set identification: each predicted face queries a gallery of all true faces.

    Distance = mean symmetric closest-point distance in the face ROI after an
    optional rigid ICP (removes residual pose differences). Returns rank-k
    accuracy, the cumulative match curve, and the full distance matrix.
    """
    names = sorted(set(preds) & set(gts))
    P = {k: _roi_points(preds[k], roi, n_points, 1) for k in names}
    G = {k: _roi_points(gts[k], roi, n_points, 2) for k in names}
    trees = {k: cKDTree(G[k]) for k in names}
    D = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            p = P[a]
            if icp:
                # Rigid alignment only; scale is identity information and must be kept.
                T, p, _ = trimesh.registration.icp(p, G[b], max_iterations=30, reflection=False,
                                                    scale=False, translation=True)
            d1 = trees[b].query(p)[0].mean()
            d2 = cKDTree(p).query(G[b])[0].mean()
            D[i, j] = 0.5 * (d1 + d2)
    ranks = np.array([(D[i] < D[i, i]).sum() + 1 for i in range(len(names))])
    n = len(names)
    cmc = [float((ranks <= k).mean()) for k in range(1, n + 1)]
    return dict(names=names, distance=D.tolist(), ranks=ranks.tolist(), rank1=cmc[0],
                rank5=cmc[min(4, n - 1)], cmc=cmc, chance_rank1=1.0 / n, gallery_size=n)
