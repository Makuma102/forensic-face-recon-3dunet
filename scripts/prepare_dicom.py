"""Screen and preprocess any local folder of DICOM studies (NMDID, TCIA, hospital data).

Each immediate sub-folder of --src is treated as one subject (put one study per folder).
Writes one .npz per subject that passes QC and a screening.csv with every decision.

    python scripts/prepare_dicom.py --src /path/to/nmdid_download --out data/nmdid --workers 4
    python scripts/prepare_dicom.py --src ... --keep-top-mm 330   # whole-body scans: keep the head end
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ffr.io_dicom import load_series  # noqa: E402
from ffr.preprocess import PrepConfig, make_sample  # noqa: E402
from ffr.screening import choose_series  # noqa: E402

FIELDS = ["case", "status", "reason", "series", "slice_mm", "extent_mm", "kernel", "tilt_deg", "sex", "age",
          "vertex_to_bottom_mm", "face_fov_clip_fraction", "lateral_body_ml", "seconds"]


def process(study: str, out: str, keep_top_mm: float | None, max_slice_mm: float) -> dict:
    t0 = time.time()
    case = os.path.basename(study.rstrip("/"))
    row = dict(case=case, status="rejected", reason="")
    cfg = PrepConfig()
    try:
        files, s = choose_series(study, max_slice_mm=max_slice_mm,
                                 min_extent_mm=0 if keep_top_mm else cfg.min_vertex_to_bottom_mm)
        if files is None:
            row["reason"] = s
            return row
        row.update(series=s["description"], slice_mm=round(s["slice_mm"], 3), extent_mm=round(s["extent_mm"], 1),
                   kernel=s["kernel"])
        vol = load_series(files, cfg.work_spacing, keep_top_mm=keep_top_mm)
        row.update(tilt_deg=round(vol.meta["tilt_deg_from_ipp"], 2), sex=vol.meta["patient_sex"],
                   age=vol.meta["patient_age"])
        sample, qc = make_sample(vol, cfg)
        row.update({k: round(v, 4) for k, v in qc.values.items() if k in FIELDS})
        if sample is None:
            row["reason"] = "; ".join(qc.reasons)
            return row
        arrays = {k: v for k, v in sample.items() if isinstance(v, np.ndarray)}
        np.savez_compressed(os.path.join(out, case + ".npz"), **arrays,
                            meta=json.dumps(dict(case=case, anchors=sample["anchors"], qc=qc.values,
                                                 series=row["series"], kernel=row["kernel"], tilt_deg=row["tilt_deg"],
                                                 sex=row["sex"], age=row["age"], source=os.path.abspath(study),
                                                 work_spacing=sample["work_spacing"],
                                                 out_spacing=sample["out_spacing"])))
        row["status"] = "ok"
        return row
    except Exception as e:
        row.update(status="error", reason=repr(e)[:200])
        return row
    finally:
        row["seconds"] = round(time.time() - t0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--keep-top-mm", type=float, default=None)
    ap.add_argument("--max-slice-mm", type=float, default=2.5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    studies = sorted(os.path.join(a.src, d) for d in os.listdir(a.src) if os.path.isdir(os.path.join(a.src, d)))
    log = os.path.join(a.out, "screening.csv")
    done = set()
    if os.path.exists(log):
        with open(log) as f:
            done = {r["case"] for r in csv.DictReader(f)}
    todo = [s for s in studies if os.path.basename(s) not in done]
    print(f"{len(studies)} studies, {len(todo)} to process", flush=True)
    new = not os.path.exists(log)
    n = len(todo)
    with open(log, "a", newline="") as f, ProcessPoolExecutor(a.workers) as ex:
        w = csv.DictWriter(f, FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        for row in ex.map(process, todo, [a.out] * n, [a.keep_top_mm] * n, [a.max_slice_mm] * n):
            w.writerow(row)
            f.flush()
            print(row["case"], row["status"], row["reason"], flush=True)


if __name__ == "__main__":
    main()
