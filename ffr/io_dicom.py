"""DICOM series loading with gantry-tilt correction.

Head CT is often acquired with a tilted gantry. Most readers stack the slices
on an orthogonal grid, which shears the anatomy and biases every downstream
distance measured in mm. Here the voxel->world affine is built from the real
ImagePositionPatient of every slice, so a tilt shows up as a non-orthogonal
slice axis, and the volume is then resampled onto an isotropic world grid.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pydicom
from scipy import ndimage


@dataclass
class Volume:
    array: np.ndarray        # (z, y, x) float32 HU, isotropic
    spacing: float           # mm, isotropic
    origin: np.ndarray       # world position (x, y, z) of voxel [0, 0, 0], LPS mm
    meta: dict


def _is_dicom(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except OSError:
        return False


def list_series(root: str) -> dict[str, list[str]]:
    """Group the DICOM files below ``root`` by SeriesInstanceUID."""
    series: dict[str, list[str]] = {}
    for dirpath, _, files in os.walk(root):
        for name in files:
            p = os.path.join(dirpath, name)
            if not (name.lower().endswith(".dcm") or _is_dicom(p)):
                continue
            try:
                ds = pydicom.dcmread(p, stop_before_pixels=True)
            except Exception:
                continue
            series.setdefault(str(ds.SeriesInstanceUID), []).append(p)
    return series


def _read_slices(files: list[str], decode: bool = True):
    slices = []
    for p in files:
        ds = pydicom.dcmread(p, stop_before_pixels=not decode)
        ds.filename = p
        if not hasattr(ds, "ImagePositionPatient"):
            continue
        slices.append(ds)
    if len(slices) < 2:
        raise ValueError("series has fewer than two image slices")
    iop = np.asarray(slices[0].ImageOrientationPatient, dtype=np.float64)
    row, col = iop[:3], iop[3:]
    normal = np.cross(row, col)
    slices.sort(key=lambda s: float(np.dot(normal, np.asarray(s.ImagePositionPatient, float))))
    # Drop duplicated positions (some series repeat a slice).
    uniq, last = [], None
    for s in slices:
        pos = float(np.dot(normal, np.asarray(s.ImagePositionPatient, float)))
        if last is None or abs(pos - last) > 1e-3:
            uniq.append(s)
            last = pos
    return uniq, row, col


def series_affine(slices, row, col) -> tuple[np.ndarray, dict]:
    """4x4 affine mapping voxel (x=i column, y=j row, z=k slice) to world LPS mm."""
    ipp = np.stack([np.asarray(s.ImagePositionPatient, float) for s in slices])
    k = np.arange(len(slices), dtype=np.float64)
    # Least-squares line through the slice origins; its slope is the slice
    # step vector, which carries the gantry tilt as an in-plane shift.
    A = np.stack([np.ones_like(k), k], 1)
    coef, res, *_ = np.linalg.lstsq(A, ipp, rcond=None)
    origin, step = coef[0], coef[1]
    resid = np.abs(A @ coef - ipp).max()
    gaps = np.diff(ipp @ np.cross(row, col))
    ps = np.asarray(slices[0].PixelSpacing, float)  # (row spacing, col spacing)
    aff = np.eye(4)
    aff[:3, 0] = row * ps[1]   # moving along a row = increasing column index
    aff[:3, 1] = col * ps[0]
    aff[:3, 2] = step
    aff[:3, 3] = origin
    normal = np.cross(row, col)
    slice_gap = float(np.dot(step, normal))
    shear = step - slice_gap * normal
    tilt_deg = float(np.degrees(np.arctan2(np.linalg.norm(shear), abs(slice_gap))))
    info = dict(n_slices=len(slices), slice_gap_mm=abs(slice_gap), tilt_deg_from_ipp=tilt_deg,
                pixel_spacing=ps.tolist(), max_position_residual_mm=float(resid),
                max_gap_ratio=float(gaps.max() / np.median(gaps)))
    return aff, info


def load_series(files: list[str], spacing: float = 1.0, order: int = 1,
                keep_top_mm: float | None = None) -> Volume:
    """Read one series, correct tilt, and resample to an isotropic world grid.

    keep_top_mm: keep only slices within this distance of the most superior one
    (crops long whole-body scans to the head before any pixels are decoded).
    """
    slices, row, col = _read_slices(files, decode=False)
    if keep_top_mm is not None:
        zs = np.array([float(s.ImagePositionPatient[2]) for s in slices])
        slices = [s for s, z in zip(slices, zs) if z >= zs.max() - keep_top_mm]
    slices = [pydicom.dcmread(s.filename) for s in slices]
    vol = np.stack([s.pixel_array.astype(np.float32) * float(getattr(s, "RescaleSlope", 1))
                    + float(getattr(s, "RescaleIntercept", 0)) for s in slices])  # (k, j, i)
    aff, info = series_affine(slices, row, col)
    if info["max_gap_ratio"] > 1.5:
        raise ValueError(f"missing slices (largest gap {info['max_gap_ratio']:.1f}x the median)")
    if info["max_position_residual_mm"] > 0.5 * info["slice_gap_mm"]:
        raise ValueError(f"irregular slice positions (residual {info['max_position_residual_mm']:.2f} mm)")

    # Bounding box of the acquired volume in world space.
    nk, nj, ni = vol.shape
    corners = np.array([[i, j, k, 1] for i in (0, ni - 1) for j in (0, nj - 1) for k in (0, nk - 1)], float)
    wc = (aff @ corners.T).T[:, :3]
    lo, hi = wc.min(0), wc.max(0)
    shape_xyz = np.floor((hi - lo) / spacing).astype(int) + 1

    # Output voxel (z,y,x) -> world -> input voxel (k,j,i).
    out2world = np.eye(4)
    out2world[:3, :3] = np.diag([spacing] * 3)[:, ::-1]  # columns ordered z, y, x
    out2world[:3, 3] = lo
    world2in = np.linalg.inv(aff)
    m = world2in @ out2world               # maps (z,y,x,1) -> (i,j,k,1)
    m = m[[2, 1, 0, 3]]                    # reorder rows to (k,j,i)
    out = ndimage.affine_transform(vol, m[:3, :3], offset=m[:3, 3],
                                   output_shape=tuple(shape_xyz[::-1]), order=order,
                                   cval=-1024.0, mode="constant")
    s0 = slices[0]
    meta = dict(info,
                gantry_tilt_tag=float(getattr(s0, "GantryDetectorTilt", 0) or 0),
                kernel=str(getattr(s0, "ConvolutionKernel", "")),
                series_description=str(getattr(s0, "SeriesDescription", "")),
                manufacturer=str(getattr(s0, "Manufacturer", "")),
                patient_sex=str(getattr(s0, "PatientSex", "")),
                patient_age=str(getattr(s0, "PatientAge", "")),
                orientation=np.round(np.concatenate([row, col]), 3).tolist())
    return Volume(out.astype(np.float32), float(spacing), lo, meta)
