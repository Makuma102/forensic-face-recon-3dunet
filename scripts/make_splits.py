"""Subject-level K-fold splits (each subject appears in exactly one test fold).

    python scripts/make_splits.py --dirs data/cq500 data/idc --k 5 --out data/splits.json
"""
import argparse
import glob
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dirs", nargs="+", required=True)
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--val-frac", type=float, default=0.15)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--external-test", nargs="*", default=[],
                help="folders kept entirely for testing (e.g. a different source), never trained on")
ap.add_argument("--out", default="data/splits.json")
a = ap.parse_args()

files = sorted(f for d in a.dirs for f in glob.glob(os.path.join(d, "*.npz")) if d not in a.external_test)
ext = sorted(f for d in a.external_test for f in glob.glob(os.path.join(d, "*.npz")))
rng = np.random.default_rng(a.seed)
order = rng.permutation(len(files))
folds = np.array_split(order, a.k)
out = dict(seed=a.seed, k=a.k, external_test=ext, folds=[])
for i in range(a.k):
    test = [files[j] for j in folds[i]]
    rest = [files[j] for j in order if j not in set(folds[i])]
    n_val = max(1, int(round(a.val_frac * len(rest))))
    out["folds"].append(dict(train=rest[n_val:], val=rest[:n_val], test=test))
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
print(f"{len(files)} subjects, {a.k} folds, {len(ext)} external test")
