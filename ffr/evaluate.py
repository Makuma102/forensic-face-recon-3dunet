"""Evaluate the U-Net or a baseline on a fold's test subjects.

    python -m ffr.evaluate --method unet --ckpt runs/unet_small/fold0/best.pt --fold 0
    python -m ffr.evaluate --method ridge --fold 0

Writes per-subject surface metrics, a summary, and the identification result.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from .baselines import BASELINES
from .data import coord_channels, load_meta
from .metrics import identification, sdf_to_mesh, surface_metrics, upsample_to
from .model import UNet3D


def sliding_window(model, inp: torch.Tensor, patch, overlap: float = 0.5) -> torch.Tensor:
    """Patch-wise inference with Gaussian blending."""
    shape = inp.shape[2:]
    ps = [min(p, n) for p, n in zip(patch, shape)]
    starts = []
    for n, p in zip(shape, ps):
        step = max(1, int(p * (1 - overlap)))
        s = list(range(0, n - p + 1, step))
        if s[-1] != n - p:
            s.append(n - p)
        starts.append(s)
    g = [np.exp(-0.5 * ((np.arange(p) - (p - 1) / 2) / (p / 4)) ** 2) for p in ps]
    w = torch.from_numpy(np.einsum("i,j,k->ijk", *g).astype(np.float32))
    out = torch.zeros(shape)
    acc = torch.zeros(shape)
    for z in starts[0]:
        for y in starts[1]:
            for x in starts[2]:
                sl = (slice(z, z + ps[0]), slice(y, y + ps[1]), slice(x, x + ps[2]))
                out[sl] += model(inp[(slice(None), slice(None), *sl)])[0, 0] * w
                acc[sl] += w
    return (out / acc)[None, None]


def menton_mm(skull_fine: np.ndarray, sp: float, front_mm: float = 40.0, depth_mm: float = 35.0) -> float | None:
    """Height (crop mm) of the lowest point of the chin: lowest bone in the front 35 mm of the skull.

    Derived from the skull (the model input), so using it for the ROI leaks nothing.
    """
    y1 = int((front_mm + depth_mm) / sp)
    zz = np.flatnonzero(skull_fine[:, :y1].any((1, 2)))
    return float(zz.min() * sp) if len(zz) else None


def roi_from_meta(meta: dict, fine_shape, sp: float, depth_mm: float, chin_mm: float | None = None) -> dict:
    """Anterior face region in crop mm: front 'depth_mm' behind the anterior-most bone, chin to brow."""
    size = np.array(fine_shape) * sp
    front = 40.0  # PrepConfig.front_margin_mm: the anterior-most bone sits 40 mm behind the crop front
    vertex = size[0] - 4.0 - sp  # vertex sits ~4 mm below the crop top (see make_sample)
    z_min = chin_mm - 5.0 if chin_mm is not None else max(0.0, vertex - 235.0)
    return dict(y_max_mm=front + depth_mm, z_min_mm=max(0.0, z_min), z_max_mm=vertex - 60.0,
                border_mm=3.0, size_mm=tuple(size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["unet", *BASELINES])
    ap.add_argument("--ckpt")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--subset", default="test", choices=["test", "val", "external_test"])
    ap.add_argument("--roi-depth-mm", type=float, default=80.0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--save-meshes", action="store_true")
    a = ap.parse_args()

    splits = json.load(open(a.splits))
    fold = splits["folds"][a.fold]
    files = splits["external_test"] if a.subset == "external_test" else fold[a.subset]
    out = os.path.join(a.out, a.method, f"fold{a.fold}_{a.subset}")
    os.makedirs(out, exist_ok=True)

    if a.method == "unet":
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        m = cfg["model"]
        model = UNet3D(in_channels=4 if cfg["data"]["coords"] else 1, base=m["base"], depth=m["depth"],
                       norm=m["norm"], residual=m["residual"], skip_connections=m["skip_connections"],
                       dropout=m.get("dropout", 0.0), predict_residual=m["predict_residual"])
        model.load_state_dict(ck["model"])
        model.eval()
        T = cfg["data"]["trunc_mm"]

        patch = cfg["data"].get("patch")

        def predict(skull, sp):
            x = skull / T
            inp = x[None]
            if cfg["data"]["coords"]:
                inp = np.concatenate([inp, coord_channels(x.shape)], 0)
            inp = torch.from_numpy(inp[None]).float()
            with torch.no_grad():
                if patch:
                    # Same patch size as in training, so instance-norm statistics match.
                    return sliding_window(model, inp, patch)[0, 0].numpy() * T
                return model(inp)[0, 0].numpy() * T
    else:
        base = BASELINES[a.method]()
        base.fit(fold["train"] + fold["val"])
        predict = base.predict

    rows, preds, gts = [], {}, {}
    for f in files:
        meta = load_meta(f)
        with np.load(f) as z:
            skull = z["skull_sdf"].astype(np.float32)
            gt_fine = z["skin_sdf_fine"].astype(np.float32)
            valid = z["valid"]
            chin = menton_mm(z["skull_mask_fine"], meta["work_spacing"])
        sp_out, sp_fine = meta["out_spacing"], meta["work_spacing"]
        pred = predict(skull, sp_out)
        pred_fine = upsample_to(pred, gt_fine.shape)
        valid_fine = upsample_to(valid.astype(np.float32), gt_fine.shape) > 0.5
        pm, gm = sdf_to_mesh(pred_fine, sp_fine), sdf_to_mesh(gt_fine, sp_fine)
        roi = roi_from_meta(meta, gt_fine.shape, sp_fine, a.roi_depth_mm, chin)
        if pm is None:
            rows.append(dict(case=meta["case"], failed=True))
            continue
        r = surface_metrics(pm, gm, roi, gt_sdf_fine=gt_fine, spacing_fine=sp_fine, valid_fine=valid_fine)
        r.update(case=meta["case"], sex=meta.get("sex"), age=meta.get("age"))
        rows.append(r)
        preds[meta["case"]], gts[meta["case"]] = pm, gm
        if a.save_meshes:
            pm.export(os.path.join(out, meta["case"] + "_pred.ply"))
            gm.export(os.path.join(out, meta["case"] + "_gt.ply"))
        print(meta["case"], {k: round(v, 2) for k, v in r.items() if isinstance(v, float)}, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "per_subject.csv"), index=False)
    num = df.select_dtypes("number")
    summary = dict(method=a.method, fold=a.fold, subset=a.subset, n=len(df),
                   mean=num.mean().to_dict(), std=num.std().to_dict())
    if len(preds) >= 2:
        ident = identification(preds, gts, roi)
        summary["identification"] = {k: ident[k] for k in ("rank1", "rank5", "chance_rank1", "gallery_size")}
        json.dump(ident, open(os.path.join(out, "identification.json"), "w"))
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(json.dumps({k: summary[k] for k in summary if k != "std"}, indent=1, default=float))


if __name__ == "__main__":
    main()
