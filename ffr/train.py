"""Train the 3D U-Net on one fold.

    python -m ffr.train --config configs/unet_small.yaml [--set train.epochs=50 data.fold=1]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data import SkullFaceDataset
from .losses import sdf_loss
from .model import UNet3D, count_parameters


def load_config(path: str, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(open(path))
    for o in overrides:
        key, val = o.split("=", 1)
        d = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            d = d[p]
        d[parts[-1]] = yaml.safe_load(val)
    return cfg


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(cfg: dict) -> UNet3D:
    m = cfg["model"]
    return UNet3D(in_channels=4 if cfg["data"]["coords"] else 1, base=m["base"], depth=m["depth"],
                  norm=m["norm"], residual=m["residual"], skip_connections=m["skip_connections"],
                  dropout=m.get("dropout", 0.0), predict_residual=m["predict_residual"])


@torch.no_grad()
def validate(model, loader, cfg, device) -> float:
    model.eval()
    T = cfg["data"]["trunc_mm"]
    errs = []
    for x, y, v, _ in loader:
        x, y, v = x.to(device), y.to(device), v.to(device)
        p = model(x)
        near = (y.abs() < 3.0 / T) & (v > 0)   # within 3 mm of the true skin
        errs.append(((p - y).abs()[near].mean() * T).item())
    return float(np.mean(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    seed_everything(cfg["seed"])
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    splits = json.load(open(cfg["data"]["splits"]))
    fold = splits["folds"][cfg["data"]["fold"]]
    dcfg = cfg["data"]
    tr = SkullFaceDataset(fold["train"], dcfg["trunc_mm"], dcfg["coords"], dcfg["augment"], seed=cfg["seed"],
                          patch=dcfg.get("patch"))
    va = SkullFaceDataset(fold["val"], dcfg["trunc_mm"], dcfg["coords"], False)
    # With patches, each epoch draws several patches per subject.
    reps = cfg["train"].get("patches_per_subject", 1) if dcfg.get("patch") else 1
    tr_ep = torch.utils.data.ConcatDataset([tr] * reps)
    tl = DataLoader(tr_ep, batch_size=cfg["train"]["batch_size"], shuffle=True,
                    num_workers=cfg["train"]["num_workers"], persistent_workers=cfg["train"]["num_workers"] > 0)
    vl = DataLoader(va, batch_size=1, num_workers=cfg["train"]["num_workers"])

    model = build_model(cfg).to(device)
    out = os.path.join(cfg["train"]["out_dir"], f"fold{dcfg['fold']}")
    os.makedirs(out, exist_ok=True)
    yaml.safe_dump(cfg, open(os.path.join(out, "config.yaml"), "w"))
    print(f"params={count_parameters(model):,} train={len(tr)} val={len(va)} device={device}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    E = cfg["train"]["epochs"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["train"]["lr"], total_steps=E * len(tl), pct_start=0.05)
    L = cfg["loss"]
    best = np.inf
    log = open(os.path.join(out, "log.csv"), "w", newline="")
    w = csv.writer(log)
    w.writerow(["epoch", "train_loss", "val_near_surface_mae_mm", "seconds"])
    for ep in range(1, E + 1):
        model.train()
        t0, losses = time.time(), []
        for x, y, v, _ in tl:
            x, y, v = x.to(device), y.to(device), v.to(device)
            loss = sdf_loss(model(x), y, v, L["surface_weight"], L["sigma"], L["sign_weight"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        val = np.nan
        if ep % cfg["train"]["val_every"] == 0 or ep == E:
            val = validate(model, vl, cfg, device)
            if val < best:
                best = val
                torch.save(dict(model=model.state_dict(), cfg=cfg, epoch=ep, val=val), os.path.join(out, "best.pt"))
        torch.save(dict(model=model.state_dict(), cfg=cfg, epoch=ep), os.path.join(out, "last.pt"))
        w.writerow([ep, float(np.mean(losses)), val, round(time.time() - t0, 1)])
        log.flush()
        print(f"epoch {ep} loss {np.mean(losses):.4f} val_mae_mm {val:.3f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
