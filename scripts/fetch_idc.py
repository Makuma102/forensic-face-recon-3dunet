"""Screen open CT series from the NCI Imaging Data Commons (IDC) for full-face head coverage.

IDC mirrors public TCIA collections in open cloud buckets; ``idc-index`` ships
the metadata. Candidates are CT series whose body part mentions the head/neck
or whole body, with thin regular slices and >= 240 mm of coverage. Each series
is downloaded, screened with the same QC as CQ500, preprocessed, and deleted.

    python scripts/fetch_idc.py --out data/idc --workers 2
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
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ffr.io_dicom import load_series  # noqa: E402
from ffr.preprocess import PrepConfig, make_sample  # noqa: E402

FIELDS = ["case", "status", "reason", "collection", "license", "series_uid", "series", "slice_mm",
          "extent_mm", "sex", "age", "tilt_deg", "vertex_to_bottom_mm", "face_fov_clip_fraction", "seconds"]


def candidates(max_slice_mm: float, min_extent_mm: float):
    from idc_index import IDCClient
    c = IDCClient()
    c.fetch_index("volume_geometry_index")
    ct = c.index[c.index.Modality == "CT"].merge(c.volume_geometry_index, on="SeriesInstanceUID")
    ct["extent"] = ct.instanceCount * ct.slice_spacing_mm
    sel = ct[(ct.regularly_spaced_3d_volume == True) & (ct.slice_spacing_mm <= max_slice_mm)  # noqa: E712
             & (ct.extent >= min_extent_mm) & (ct.obliquity_degrees < 1)
             & ct.BodyPartExamined.astype(str).str.contains("HEAD|WHOLE|NECK|BRAIN|SKULL", case=False)]
    # One series per patient: the thinnest, then the longest.
    sel = sel.sort_values(["PatientID", "slice_spacing_mm", "extent"], ascending=[True, True, False])
    return sel.drop_duplicates("PatientID")


def process(row: dict, out: str) -> dict:
    t0 = time.time()
    case = f"IDC-{row['collection_id']}-{row['PatientID']}".replace("/", "_")
    rec = dict(case=case, status="rejected", reason="", collection=row["collection_id"],
               license=row["license_short_name"], series_uid=row["SeriesInstanceUID"],
               series=row["SeriesDescription"], slice_mm=row["slice_spacing_mm"], extent_mm=round(row["extent"], 1))
    tmp = tempfile.mkdtemp(prefix="idc_")
    cfg = PrepConfig()
    try:
        from idc_index import IDCClient
        IDCClient().download_from_selection(seriesInstanceUID=[row["SeriesInstanceUID"]], downloadDir=tmp,
                                            show_progress_bar=False, quiet=True)
        files = [os.path.join(d, f) for d, _, fs in os.walk(tmp) for f in fs if f.endswith(".dcm")]
        vol = load_series(files, cfg.work_spacing, keep_top_mm=330.0)
        rec.update(sex=vol.meta["patient_sex"], age=vol.meta["patient_age"],
                   tilt_deg=round(vol.meta["tilt_deg_from_ipp"], 2))
        sample, qc = make_sample(vol, cfg)
        rec.update({k: round(v, 4) for k, v in qc.values.items() if k in FIELDS})
        if sample is None:
            rec["reason"] = "; ".join(qc.reasons)
            return rec
        arrays = {k: v for k, v in sample.items() if isinstance(v, np.ndarray)}
        np.savez_compressed(os.path.join(out, case + ".npz"), **arrays,
                            meta=json.dumps(dict(case=case, anchors=sample["anchors"], qc=qc.values,
                                                 collection=row["collection_id"], license=row["license_short_name"],
                                                 series_uid=row["SeriesInstanceUID"], sex=rec["sex"], age=rec["age"],
                                                 work_spacing=sample["work_spacing"],
                                                 out_spacing=sample["out_spacing"])))
        rec["status"] = "ok"
        return rec
    except Exception as e:
        rec.update(status="error", reason=repr(e)[:200])
        return rec
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        rec["seconds"] = round(time.time() - t0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/idc")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-slice-mm", type=float, default=3.0)
    ap.add_argument("--min-extent-mm", type=float, default=240.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sel = candidates(a.max_slice_mm, a.min_extent_mm)
    log = os.path.join(a.out, "screening.csv")
    done = set()
    if os.path.exists(log):
        with open(log) as f:
            done = {r["series_uid"] for r in csv.DictReader(f)}
    rows = [r for r in sel.to_dict("records") if r["SeriesInstanceUID"] not in done]
    print(f"{len(sel)} candidate series, {len(rows)} to screen", flush=True)
    new = not os.path.exists(log)
    with open(log, "a", newline="") as f, ProcessPoolExecutor(a.workers) as ex:
        w = csv.DictWriter(f, FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        for rec in ex.map(process, rows, [a.out] * len(rows)):
            w.writerow(rec)
            f.flush()
            print(rec["case"], rec["status"], rec["reason"], flush=True)


if __name__ == "__main__":
    main()
