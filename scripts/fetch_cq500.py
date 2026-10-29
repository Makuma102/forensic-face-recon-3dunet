"""Download CQ500 studies one at a time, keep only those whose CT covers the whole face.

CQ500 (Chilamkurthy et al., Lancet 2018) is a public head-CT set of 491 studies.
Most are routine brain CTs that stop at the skull base, so only a minority are
usable for skull->face work. Each zip is deleted after screening, and only the
preprocessed sample plus a QC row are kept.

    python scripts/fetch_cq500.py --out data/cq500 --start 0 --stop 491 --workers 3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ffr.io_dicom import load_series  # noqa: E402
from ffr.preprocess import PrepConfig, make_sample  # noqa: E402
from ffr.screening import choose_series  # noqa: E402

URL = "https://s3.ap-south-1.amazonaws.com/qure.headct.study/CQ500-CT-{i}.zip"
FIELDS = ["case", "status", "reason", "series", "slice_mm", "extent_mm", "kernel",
          "tilt_deg", "sex", "age", "vertex_to_bottom_mm", "face_fov_clip_fraction", "seconds"]


def process(i: int, out: str, keep_zip_dir: str | None) -> dict:
    t0 = time.time()
    case = f"CQ500-CT-{i}"
    row = dict(case=case, status="rejected", reason="")
    cfg = PrepConfig()
    tmp = tempfile.mkdtemp(prefix=f"cq{i}_")
    try:
        zpath = os.path.join(tmp, "s.zip")
        for attempt in range(4):
            try:
                urllib.request.urlretrieve(URL.format(i=i), zpath)
                break
            except Exception as e:  # network hiccup
                if attempt == 3:
                    row.update(status="error", reason=f"download: {e}")
                    return row
                time.sleep(2 ** (attempt + 1))
        with zipfile.ZipFile(zpath) as z:
            z.extractall(os.path.join(tmp, "x"))
        if keep_zip_dir:
            shutil.copy(zpath, os.path.join(keep_zip_dir, case + ".zip"))
        os.remove(zpath)
        files, s = choose_series(os.path.join(tmp, "x"), min_extent_mm=cfg.min_vertex_to_bottom_mm)
        if files is None:
            row["reason"] = s
            return row
        row.update(series=s["description"], slice_mm=round(s["slice_mm"], 3),
                   extent_mm=round(s["extent_mm"], 1), kernel=s["kernel"])
        vol = load_series(files, cfg.work_spacing)
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
                                                 series=row["series"], kernel=row["kernel"],
                                                 tilt_deg=row["tilt_deg"], sex=row["sex"], age=row["age"],
                                                 work_spacing=sample["work_spacing"],
                                                 out_spacing=sample["out_spacing"])))
        row["status"] = "ok"
        return row
    except Exception as e:
        row.update(status="error", reason=repr(e)[:200])
        return row
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        row["seconds"] = round(time.time() - t0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/cq500")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=491)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--keep-zips", default=None, help="optional folder to keep the raw zips")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = os.path.join(a.out, "screening.csv")
    done = set()
    if os.path.exists(log):
        with open(log) as f:
            done = {r["case"] for r in csv.DictReader(f)}
    todo = [i for i in range(a.start, a.stop) if f"CQ500-CT-{i}" not in done]
    new = not os.path.exists(log)
    with open(log, "a", newline="") as f, ProcessPoolExecutor(a.workers) as ex:
        w = csv.DictWriter(f, FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        for row in ex.map(process, todo, [a.out] * len(todo), [a.keep_zips] * len(todo)):
            w.writerow(row)
            f.flush()
            print(row["case"], row["status"], row.get("reason", ""), flush=True)


if __name__ == "__main__":
    main()
