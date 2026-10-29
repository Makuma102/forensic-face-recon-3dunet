"""Visual QC sheet: mid-sagittal skull and skin for every preprocessed subject.

Look at every subject before training. Reject scans with a thermoplastic mask,
open mouth / bite block, nasal tubes, heavy dental artefacts, severe facial
trauma or surgery, or a face cut by the field of view.

    python scripts/qc_montage.py --dirs data/cq500 data/idc --out qc.png
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("--dirs", nargs="+", required=True)
ap.add_argument("--out", default="qc.png")
ap.add_argument("--cols", type=int, default=6)
a = ap.parse_args()

files = sorted(f for d in a.dirs for f in glob.glob(os.path.join(d, "*.npz")))
tiles = []
for f in files:
    with np.load(f) as z:
        s = z["skull_sdf"].astype(np.float32)
        t = z["skin_sdf"].astype(np.float32)
    x = s.shape[2] // 2
    sag = np.full(s[:, :, x].shape + (3,), 0, np.uint8)
    sag[t[:, :, x] < 0] = (90, 70, 60)
    sag[s[:, :, x] < 0] = (240, 240, 230)
    # Axial slice through the nose (most anterior skin at mid-height).
    zmid = int(s.shape[0] * 0.45)
    ax = np.full(s[zmid].shape + (3,), 0, np.uint8)
    ax[t[zmid] < 0] = (90, 70, 60)
    ax[s[zmid] < 0] = (240, 240, 230)
    tile = np.concatenate([sag[::-1], np.zeros((sag.shape[0], 4, 3), np.uint8),
                           np.pad(ax, ((0, sag.shape[0] - ax.shape[0]), (0, 0), (0, 0)))], 1)
    im = Image.fromarray(tile).resize((tile.shape[1] * 2, tile.shape[0] * 2), Image.NEAREST)
    ImageDraw.Draw(im).text((4, 4), os.path.basename(f)[:-4][:30], fill=(255, 255, 0))
    tiles.append(im)
if not tiles:
    raise SystemExit("no samples")
W, H = tiles[0].size
rows = (len(tiles) + a.cols - 1) // a.cols
sheet = Image.new("RGB", (W * min(a.cols, len(tiles)), H * rows))
for i, im in enumerate(tiles):
    sheet.paste(im, ((i % a.cols) * W, (i // a.cols) * H))
sheet.save(a.out)
print(f"{len(tiles)} subjects -> {a.out}")
