# Experiment and ablation plan

All numbers in the paper must come from `ffr.evaluate` runs on real CT. Nothing is simulated.

## 1. Task definition

- **Input:** skull only, as a truncated signed distance field (SDF, ±20 mm) on a 2 mm grid, 128 × 112 × 96 voxels, cropped with anchors computed from the skull (vertex, anterior-most bone, mid-sagittal line). No information about the face is used to place the crop.
- **Output:** skin SDF on the same grid. The face is its zero level set (marching cubes).
- **Ground truth:** skin surface segmented from the same CT at 1 mm.

## 2. Datasets and splits

| Set | Role |
|---|---|
| NMDID (after approval), e.g. 500–1,000 adults | Main train/val/test, 5-fold cross-validation by subject |
| CQ500 + IDC full-face subset (open) | External test set (different scanners, population, living patients) and public benchmark |
| Own hospital data (optional) | Population-specific test or fine-tuning |

`scripts/make_splits.py --external-test data/cq500 data/idc` keeps the open data out of training.

## 3. Methods compared

| Method | Type |
|---|---|
| Mean face (`mean_face`) | Population template, ignores the skull |
| Constant tissue depth (`tissue_depth`) | Skull envelope offset by one depth fitted on train |
| Nearest neighbour (`nearest_neighbour`) | Face of the most similar training skull |
| Linear skull→face model (`ridge`) | Statistical (PCA/ridge) family |
| **3D U-Net (ours)** | `configs/unet_small.yaml`, ~1.4 M parameters |
| Published method(s) | Re-implement or ask authors for code (IJLM 2026 skull-conditioned U-Net; a 2D projection method) |

## 4. Metrics

Within the anterior-face region (front 80 mm of the head, chin to brow):
- symmetric surface distance: mean, median, RMS, 95th percentile, HD95 (mm)
- % of surface within 2 mm and 5 mm
- signed error (over/under-estimation of tissue thickness)
- error by facial third (upper/middle/lower); add landmark-based regions (nose, lips, eyes, cheeks) once landmarks are annotated
- **identification:** gallery of true faces, query = reconstruction, rigid ICP then mean distance; rank-1, rank-5, full CMC, chance level 1/N; repeat with larger distractor galleries

Statistics: mean ± SD over folds and subjects; per-subject paired Wilcoxon signed-rank tests vs. each baseline with Holm correction; 95 % bootstrap CIs (subjects resampled) for every headline number.

## 5. Ablations (one change at a time, same folds and seeds)

| # | Change | Config key |
|---|---|---|
| A1 | No skip connections (plain encoder–decoder) | `model.skip_connections=false` |
| A2 | Plain L1 (no surface weighting) | `loss.surface_weight=0` |
| A3 | No CoordConv channels | `data.coords=false` |
| A4 | Predict skull SDF + offset (residual) | `model.predict_residual=true` |
| A5 | Width 8 / 16 / 32 | `model.base=…` |
| A6 | No augmentation | `data.augment=false` |
| A7 | Input = binary skull mask instead of SDF | change `input_key` / preprocessing |
| A8 | Grid 2 mm vs 1.5 mm | `PrepConfig.out_spacing` |
| A9 | Training-set size 25 / 50 / 100 / all % (learning curve) | subsample `train` |
| A10 | Sex/age conditioning (FiLM) | to add |
| A11 | Ensemble of 5 / MC dropout → uncertainty map, error-vs-uncertainty calibration | to add |

Run 3 seeds for the main model and the most important ablations.

## 6. Reproducibility checklist

- Fixed `data/splits.json` committed (paths only, no images).
- `screening.csv` for every source → exclusion flow diagram.
- Config + git commit hash saved with every run (`runs/*/config.yaml`).
- Report hardware, training time, parameter count.

## 7. Suggested paper structure

1. Introduction: identification of unknown remains; limits of manual FFR; data-driven methods; gaps (no open benchmark, little identification testing, no uncertainty).
2. Related work (see `literature_review.md`).
3. Data and QC (flow diagram).
4. Method (representation, skull-only anchoring, network, loss).
5. Experiments (baselines, metrics, ablations, external test).
6. Results (tables, error maps on consented/mean faces only, CMC curves).
7. Discussion: where it fails (nose, lips, eyes), population bias, post-mortem changes, forensic limits.
8. Ethics statement and data availability.
