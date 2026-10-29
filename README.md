# Forensic facial reconstruction with a small 3D U-Net

Predict the facial soft-tissue surface from a skull, volumetrically, to support the identification of unknown deceased persons.

- **Input:** skull from CT, as a truncated signed distance field (SDF) in a skull-anchored crop (2 mm grid, 128 × 112 × 96).
- **Output:** skin SDF; the face is its zero surface.
- **Model:** compact residual 3D U-Net (≈ 1.4 M parameters at width 16), surface-weighted L1 loss on the SDF.
- **Evaluation:** surface distance in mm in the face region, error by facial third, signed tissue error, and a closed-set identification test (rank-1/rank-5, CMC).
- **Baselines:** mean face, constant tissue depth, nearest neighbour, linear (ridge) skull→face model.

Only real CT data is used. See [`docs/data_sources.md`](docs/data_sources.md) for what is open, what needs an application, and the inclusion criteria.

## Layout

```
ffr/
  io_dicom.py     DICOM series reader with gantry-tilt correction, isotropic resampling
  preprocess.py   head/bone segmentation, cranium detection, coverage QC, skull-anchored crop, SDFs
  screening.py    choose the right series in a study
  model.py        3D U-Net (skip connections, residual blocks, CoordConv, residual-SDF option)
  losses.py       surface-weighted SDF loss
  data.py         dataset, augmentation, patch sampling
  baselines.py    non-deep baselines
  metrics.py      surface metrics (mm) and identification test
  render.py       frontal shaded and sketch-style renderings of a face (for composite work)
  train.py        training (one fold)
  evaluate.py     evaluation of the U-Net or any baseline
scripts/
  fetch_cq500.py  download + screen + preprocess CQ500 (deletes raw data after)
  fetch_idc.py    same for open CT in the NCI Imaging Data Commons
  prepare_dicom.py  same QC + preprocessing for any local DICOM folder (NMDID, TCIA, hospital)
  render_face.py  render a true or predicted face as shaded + sketch images
  qc_montage.py   visual QC sheet of every preprocessed subject
  make_splits.py  subject-level K-fold splits (+ external test sets)
configs/          unet_small.yaml (GPU, full volume), unet_pilot_cpu.yaml (patches)
docs/             literature review, data sources, experiment plan, ethics
tests/            unit tests (pytest)
```

## Quick start

```bash
pip install -r requirements.txt
pytest -q

# 1. Data
#    local studies (one study per sub-folder), e.g. NMDID whole-body CT:
python scripts/prepare_dicom.py --src /path/to/nmdid --out data/nmdid --keep-top-mm 330 --workers 4
#    open sources:
python scripts/fetch_cq500.py --out data/cq500 --workers 3
python scripts/fetch_idc.py   --out data/idc   --workers 2
python scripts/qc_montage.py  --dirs data/cq500 data/idc --out qc.png    # inspect every subject

# 2. Splits
python scripts/make_splits.py --dirs data/cq500 data/idc --k 5 --out data/splits.json

# 3. Baselines and model
for m in mean_face tissue_depth nearest_neighbour ridge; do python -m ffr.evaluate --method $m --fold 0; done
python -m ffr.train --config configs/unet_small.yaml --set data.fold=0
python -m ffr.evaluate --method unet --ckpt runs/unet_small/fold0/best.pt --fold 0
```

For an investigator-facing image, `python scripts/render_face.py --pred pred_sdf.npy --spacing 2 --out face.png` gives a neutral shaded view and a pencil-style view. Hair, eye shape, skin colour and expression are not predictable from bone and are left to the forensic artist.

Ablations are config overrides, e.g. `--set model.skip_connections=false loss.surface_weight=0`. The full list is in [`docs/experiment_plan.md`](docs/experiment_plan.md).

## Preprocessing details

1. Read the series, sort slices along the slice normal, build the voxel→world affine from the real slice positions (this captures gantry tilt as shear) and resample to 1 mm isotropic in patient (LPS) orientation.
2. Head mask: HU > −400, 3 mm opening to detach the head holder, largest component, hole filling in 3D and slice-wise along all three axes (closes nasal and oral cavities).
3. Bone: HU > 250 inside the head, components < 0.5 ml removed.
4. Cranium detection: axial slices where a (gap-closed) bone ring encloses brain-density tissue; rejects non-head scans, and bone or tissue beside or above the head (raised arms).
5. QC: vertex present; scan reaches ≥ 235 mm below the vertex; face skin not clipped by the reconstruction field of view; no tissue beside the head.
6. Crop anchored on the skull only: top = vertex, front = anterior-most bone − 40 mm, centre = mid-sagittal bone. SDFs at 1 mm, network grid at 2 mm.

## Data and privacy

No images, meshes, or derived arrays are committed (`.gitignore`). Follow each dataset's data use agreement. See [`docs/ethics.md`](docs/ethics.md).

## License

Code: MIT. Data: governed by each source's terms.
