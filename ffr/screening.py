"""Pick a usable series from a study and run the face-coverage QC."""
from __future__ import annotations

import numpy as np
import pydicom

from .io_dicom import list_series


def series_header_summary(files: list[str]) -> dict:
    """Cheap per-series summary from headers only (no pixel decoding)."""
    pos, ds0 = [], None
    for p in files:
        ds = pydicom.dcmread(p, stop_before_pixels=True)
        if not hasattr(ds, "ImagePositionPatient"):
            continue
        ds0 = ds0 or ds
        pos.append(np.asarray(ds.ImagePositionPatient, float))
    if ds0 is None or len(pos) < 2:
        return dict(n=0)
    iop = np.asarray(ds0.ImageOrientationPatient, float)
    normal = np.cross(iop[:3], iop[3:])
    d = np.sort(np.asarray(pos) @ normal)
    gaps = np.diff(d)
    gaps = gaps[gaps > 1e-3]
    return dict(n=len(pos), extent_mm=float(d[-1] - d[0]),
                slice_mm=float(np.median(gaps)) if len(gaps) else np.inf,
                description=str(getattr(ds0, "SeriesDescription", "")),
                kernel=str(getattr(ds0, "ConvolutionKernel", "")),
                axial=bool(abs(normal[2]) > 0.8))


def choose_series(study_dir: str, max_slice_mm: float = 2.5, min_extent_mm: float = 0.0):
    """Return (files, summary) of the thinnest axial series with the largest coverage, or (None, reason)."""
    cands = []
    for uid, files in list_series(study_dir).items():
        s = series_header_summary(files)
        if s.get("n", 0) < 20 or not s["axial"]:
            continue
        s["uid"] = uid
        cands.append((files, s))
    if not cands:
        return None, "no axial series"
    thin = [c for c in cands if c[1]["slice_mm"] <= max_slice_mm]
    if not thin:
        return None, f"no series with slices <= {max_slice_mm} mm"
    # Prefer a soft-tissue kernel when several thin series cover the same range.
    def key(c):
        s = c[1]
        soft = any(k in (s["kernel"] + s["description"]).upper() for k in ("SOFT", "STANDARD", "PLAIN", "H3", "J30"))
        return (round(s["extent_mm"] / 10), soft, -s["slice_mm"])
    files, s = max(thin, key=key)
    if s["extent_mm"] < min_extent_mm:
        return None, f"longest thin series covers {s['extent_mm']:.0f} mm (< {min_extent_mm:.0f})"
    return files, s
