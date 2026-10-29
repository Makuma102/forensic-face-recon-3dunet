"""Render a predicted (or true) face as a shaded and a sketch-style frontal image.

    python scripts/render_face.py --npz data/cq500/CQ500-CT-6.npz --key skin_sdf_fine --out face.png
    python scripts/render_face.py --pred results/unet/.../CASE_pred_sdf.npy --spacing 1 --out face.png

Do not publish renderings of real subjects unless their data use terms allow it.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ffr.render import frontal_depth, shaded, sketch, to_uint8  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--npz")
ap.add_argument("--key", default="skin_sdf_fine")
ap.add_argument("--pred", help=".npy SDF in mm")
ap.add_argument("--spacing", type=float, default=None)
ap.add_argument("--out", required=True)
a = ap.parse_args()
if a.pred:
    sdf, sp = np.load(a.pred).astype(np.float32), a.spacing or 1.0
else:
    with np.load(a.npz) as z:
        sdf = z[a.key].astype(np.float32)
    sp = a.spacing or (1.0 if a.key.endswith("_fine") else 2.0)
depth, _ = frontal_depth(sdf, sp)
img = np.concatenate([shaded(depth, sp), np.ones((depth.shape[0], 6)), sketch(depth, sp)], 1)
Image.fromarray(to_uint8(img, 2 if sp <= 1 else 4)).save(a.out)
print(a.out)
