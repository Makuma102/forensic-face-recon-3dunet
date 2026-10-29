"""Collect results/<method>/fold*_<subset>/ into one table (per-subject pooled, mean ± SD).

    python scripts/summarize.py --results results --subset test
"""
import argparse
import glob
import json
import os

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--subset", default="test")
a = ap.parse_args()

cols = ["mean_mm", "rms_mm", "hd95_mm", "within_2mm", "signed_mean_mm", "lower_third_mm", "middle_third_mm",
        "upper_third_mm"]
rows = []
for mdir in sorted(glob.glob(os.path.join(a.results, "*"))):
    dfs = [pd.read_csv(f) for f in glob.glob(os.path.join(mdir, f"fold*_{a.subset}", "per_subject.csv"))]
    if not dfs:
        continue
    df = pd.concat(dfs)
    r = {"method": os.path.basename(mdir), "n": len(df)}
    for c in cols:
        if c in df:
            r[c] = f"{df[c].mean():.2f} ± {df[c].std():.2f}"
    ids = [json.load(open(f)) for f in glob.glob(os.path.join(mdir, f"fold*_{a.subset}", "summary.json"))]
    r1 = [i["identification"]["rank1"] for i in ids if "identification" in i]
    if r1:
        r["rank1 (per fold)"] = ", ".join(f"{x:.2f}" for x in r1)
    rows.append(r)
print(pd.DataFrame(rows).to_markdown(index=False))
