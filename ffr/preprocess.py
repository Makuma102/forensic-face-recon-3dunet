"""Turn a head CT volume into a paired (skull, skin) training sample.

Every step that decides *where* the crop sits uses the skull only, so nothing
about the target face leaks into the model input at test time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .io_dicom import Volume


@dataclass
class PrepConfig:
    work_spacing: float = 1.0           # mm, resolution for segmentation and GT meshes
    out_spacing: float = 2.0            # mm, network grid
    crop_zyx: tuple = (128, 112, 96)    # voxels on the network grid
    front_margin_mm: float = 40.0       # space in front of the anterior-most bone
    body_hu: float = -400.0
    bone_hu: float = 250.0
    fov_hu: float = -1500.0             # pixels below this are outside the reconstruction FOV
    min_bone_component_ml: float = 0.5
    sdf_trunc_mm: float = 20.0
    min_vertex_to_bottom_mm: float = 235.0  # scan must reach this far below the vertex
    max_fov_clip_fraction: float = 0.01


@dataclass
class QCResult:
    ok: bool
    reasons: list = field(default_factory=list)
    values: dict = field(default_factory=dict)


def _largest_cc(mask: np.ndarray) -> np.ndarray:
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum_labels(mask, lab, index=np.arange(1, n + 1))
    return lab == (np.argmax(sizes) + 1)


def _ball(radius_vox: int) -> np.ndarray:
    r = radius_vox
    z, y, x = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
    return (x * x + y * y + z * z) <= r * r


def segment_body(hu: np.ndarray, cfg: PrepConfig, spacing: float) -> np.ndarray:
    """Head/body mask: threshold, cut thin attachments (head holder, cables), keep largest part, fill."""
    m = hu > cfg.body_hu
    r = max(1, int(round(3.0 / spacing)))
    m = ndimage.binary_opening(m, structure=_ball(r))
    m = _largest_cc(m)
    m = ndimage.binary_closing(m, structure=_ball(r))
    m = ndimage.binary_fill_holes(m)
    # Slice-wise fills along all three axes close the oral and nasal cavities,
    # which are open to the outside through the nostrils or lips in 3D.
    for _ in range(2):
        for ax in range(3):
            mv = np.moveaxis(m, ax, 0)
            for i in range(mv.shape[0]):
                mv[i] = ndimage.binary_fill_holes(mv[i])
    return m


def segment_bone(hu: np.ndarray, body: np.ndarray, cfg: PrepConfig, spacing: float) -> np.ndarray:
    b = (hu > cfg.bone_hu) & body
    lab, n = ndimage.label(b)
    if n == 0:
        return b
    sizes = ndimage.sum_labels(b, lab, index=np.arange(1, n + 1)) * spacing ** 3 / 1000.0
    keep = np.flatnonzero(sizes >= cfg.min_bone_component_ml) + 1
    return np.isin(lab, keep)


def signed_distance(mask: np.ndarray, spacing: float) -> np.ndarray:
    """Euclidean signed distance in mm, negative inside, positive outside."""
    if not mask.any():
        return np.full(mask.shape, 1e3, np.float32)
    outside = ndimage.distance_transform_edt(~mask, sampling=spacing)
    inside = ndimage.distance_transform_edt(mask, sampling=spacing)
    return (outside - inside).astype(np.float32)


def find_cranium(hu: np.ndarray, bone: np.ndarray, spacing: float) -> dict | None:
    """Locate the cranial vault: axial slices where a bone ring encloses brain-density tissue.

    This stops whole-body scans from anchoring on raised arms and rejects
    neck/thorax scans that contain no head at all.
    """
    rows = []
    rr = max(1, int(round(4 / spacing)))
    yy, xx = np.ogrid[-rr:rr + 1, -rr:rr + 1]
    disk = (yy * yy + xx * xx) <= rr * rr
    for z in range(bone.shape[0]):
        b = bone[z]
        if b.sum() * spacing ** 2 < 300:  # < 3 cm^2 of bone
            continue
        # Thin temporal bone often leaves gaps in the ring at a 250 HU threshold.
        bc = ndimage.binary_closing(np.pad(b, rr), structure=disk)[rr:-rr, rr:-rr]
        filled = ndimage.binary_fill_holes(bc)
        inner = filled & ~bc
        lab, n = ndimage.label(inner)
        if n == 0:
            continue
        sizes = ndimage.sum_labels(inner, lab, index=np.arange(1, n + 1))
        k = int(np.argmax(sizes)) + 1
        area_cm2 = sizes[k - 1] * spacing ** 2 / 100
        if area_cm2 < 40:
            continue
        reg = lab == k
        med = float(np.median(hu[z][reg]))
        if -10 <= med <= 70:  # brain / CSF, not lung, fat or bowel gas
            cy, cx = np.argwhere(reg).mean(0)
            rows.append((z, area_cm2, cy, cx))
    if not rows:
        return None
    zs = np.array([r[0] for r in rows])
    # Longest run of (nearly) contiguous cranial slices.
    breaks = np.flatnonzero(np.diff(zs) > max(2, int(5 / spacing)))
    runs = np.split(np.arange(len(zs)), breaks + 1)
    run = max(runs, key=len)
    if (zs[run[-1]] - zs[run[0]]) * spacing < 35:
        return None
    sel = [rows[i] for i in run]
    w = np.array([r[1] for r in sel])
    return dict(z_lo=int(zs[run[0]]), z_hi=int(zs[run[-1]]),
                cy=float(np.average([r[2] for r in sel], weights=w)),
                cx=float(np.average([r[3] for r in sel], weights=w)),
                max_area_cm2=float(w.max()))


def head_bone(bone: np.ndarray, cran: dict, spacing: float, radius_mm: float = 110.0) -> np.ndarray:
    """Bone within a cylinder around the cranial axis, from above the vertex to 260 mm below."""
    zz, yy, xx = np.ogrid[: bone.shape[0], : bone.shape[1], : bone.shape[2]]
    cyl = ((yy - cran["cy"]) ** 2 + (xx - cran["cx"]) ** 2) * spacing ** 2 <= radius_mm ** 2
    zr = (zz >= cran["z_hi"] - int(300 / spacing))
    return bone & cyl & zr


def skull_anchor(bone: np.ndarray, spacing: float) -> dict:
    """Crop anchors from the skull alone: vertex, anterior-most bone, mid-sagittal x."""
    zz = np.flatnonzero(bone.any((1, 2)))
    vertex_z = int(zz.max())  # array z increases superiorly (LPS)
    # Anterior extent only from the cranium and face, not the cervical spine.
    upper = bone[max(0, vertex_z - int(240 / spacing)):vertex_z + 1]
    ys = np.flatnonzero(upper.any((0, 2)))
    front_y = int(ys.min())  # LPS: y grows posteriorly, so anterior = small index
    xs = np.nonzero(upper)[2]
    mid_x = float(np.median(xs))
    return dict(vertex_z=vertex_z, front_y=front_y, mid_x=mid_x)


def coverage_qc(hu: np.ndarray, body: np.ndarray, bone: np.ndarray, infov: np.ndarray,
                cfg: PrepConfig, spacing: float) -> QCResult:
    """Reject scans whose field of view cuts the face (most routine brain CTs do)."""
    q = QCResult(ok=True)
    cran = find_cranium(hu, bone, spacing)
    if cran is None:
        q.ok = False
        q.reasons.append("no cranial vault found (not a head scan, or head cut off)")
        return q
    hb = head_bone(bone, cran, spacing)
    a = skull_anchor(hb, spacing)
    if a["vertex_z"] - cran["z_hi"] > int(40 / spacing):
        q.ok = False
        q.reasons.append("bone far above the cranial vault (raised arms?)")
    # Body tissue lateral to the head (raised arms, hands, positioning aids) corrupts the skin target.
    zlo = max(0, a["vertex_z"] - int(160 / spacing))  # above the shoulders
    zz, yy, xx = np.nonzero(body[zlo:a["vertex_z"] + 1])
    far = ((yy - cran["cy"]) ** 2 + (xx - cran["cx"]) ** 2) * spacing ** 2 > 130.0 ** 2
    q.values["lateral_body_ml"] = float(far.sum() * spacing ** 3 / 1000)
    if q.values["lateral_body_ml"] > 20:
        q.ok = False
        q.reasons.append("body tissue beside the head (arms raised or attached device)")
    top_gap = (body.shape[0] - 1 - a["vertex_z"]) * spacing
    vertex_to_bottom = a["vertex_z"] * spacing
    # Body voxels that touch the edge of the reconstructed FOV are truncated skin.
    edge = infov & ~ndimage.binary_erosion(infov, structure=_ball(1))
    surf = body & ~ndimage.binary_erosion(body, structure=_ball(1))
    clip_frac = float((surf & edge).sum() / max(1, surf.sum()))
    # Only the face counts: a clipped scalp at the vertex or clipped shoulders
    # do not touch the reconstruction target.
    lo = max(0, a["vertex_z"] - int(cfg.min_vertex_to_bottom_mm / spacing))
    hi = max(lo + 1, a["vertex_z"] - int(30 / spacing))
    yb = a["front_y"] + int(120 / spacing)
    face_surf = surf[lo:hi, :yb]
    head_clip = float((face_surf & edge[lo:hi, :yb]).sum() / max(1, face_surf.sum()))
    q.values.update(top_gap_mm=top_gap, vertex_to_bottom_mm=vertex_to_bottom,
                    fov_clip_fraction=clip_frac, face_fov_clip_fraction=head_clip)
    if top_gap < spacing:
        q.ok = False
        q.reasons.append("vertex cut off")
    if vertex_to_bottom < cfg.min_vertex_to_bottom_mm:
        q.ok = False
        q.reasons.append(f"scan ends {vertex_to_bottom:.0f} mm below vertex (< {cfg.min_vertex_to_bottom_mm:.0f}); lower face missing")
    if head_clip > cfg.max_fov_clip_fraction:
        q.ok = False
        q.reasons.append(f"FOV clips {100 * head_clip:.1f}% of face skin")
    return q


def make_sample(vol: Volume, cfg: PrepConfig) -> tuple[dict | None, QCResult]:
    """Full pipeline for one CT. Returns (sample, qc); sample is None when QC fails."""
    sp = vol.spacing
    hu = vol.array
    infov = hu > cfg.fov_hu
    body = segment_body(hu, cfg, sp)
    bone = segment_bone(hu, body, cfg, sp)
    qc = coverage_qc(hu, body, bone, infov, cfg, sp)
    if not qc.ok:
        return None, qc

    bone = head_bone(bone, find_cranium(hu, bone, sp), sp)
    a = skull_anchor(bone, sp)
    f = cfg.out_spacing / sp
    cz, cy, cx = cfg.crop_zyx
    # Crop box on the fine grid, defined by skull anchors only.
    z1 = a["vertex_z"] + int(round(4 / sp)) + 1
    z0 = z1 - int(round(cz * f))
    y0 = a["front_y"] - int(round(cfg.front_margin_mm / sp))
    y1 = y0 + int(round(cy * f))
    x0 = int(round(a["mid_x"] - cx * f / 2))
    x1 = x0 + int(round(cx * f))
    box = (z0, z1, y0, y1, x0, x1)

    def crop(arr, fill):
        out = np.full((z1 - z0, y1 - y0, x1 - x0), fill, arr.dtype)
        sz = slice(max(z0, 0), min(z1, arr.shape[0]))
        sy = slice(max(y0, 0), min(y1, arr.shape[1]))
        sx = slice(max(x0, 0), min(x1, arr.shape[2]))
        out[sz.start - z0:sz.stop - z0, sy.start - y0:sy.stop - y0, sx.start - x0:sx.stop - x0] = arr[sz, sy, sx]
        return out

    bone_c, body_c, fov_c = crop(bone, False), crop(body, False), crop(infov, False)
    skull_sdf = signed_distance(bone_c, sp)
    skin_sdf = signed_distance(body_c, sp)

    def down(x, order=1):
        return ndimage.zoom(x, 1.0 / f, order=order).astype(np.float32)

    out_shape = tuple(cfg.crop_zyx)
    s_in, s_out, valid = down(skull_sdf), down(skin_sdf), down(fov_c.astype(np.float32), 0) > 0.5
    assert s_in.shape == out_shape, (s_in.shape, out_shape)
    T = cfg.sdf_trunc_mm
    sample = dict(
        skull_sdf=np.clip(s_in, -T, T).astype(np.float16),
        skin_sdf=np.clip(s_out, -T, T).astype(np.float16),
        valid=valid,                                   # voxels inside the acquired FOV
        skin_sdf_fine=np.clip(skin_sdf, -T, T).astype(np.float16),  # for evaluation meshes
        skull_mask_fine=bone_c,
        box_fine=np.array(box), anchors=a,
        work_spacing=sp, out_spacing=cfg.out_spacing,
        origin_world=vol.origin + np.array([x0, y0, z0]) * sp,
    )
    return sample, qc
