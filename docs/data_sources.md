# Data sources

The model needs CT that shows **the whole skull and the whole face** (vertex to chin, nose tip inside the field of view). Routine brain CT stops at the skull base and is unusable; head-and-neck, whole-head, or whole-body CT is needed.

## Checked here (open, downloadable without application)

| Source | Access | License | Usable for this task |
|---|---|---|---|
| **CQ500** (Chilamkurthy et al., *Lancet* 2018), 491 head CTs, India | Direct download (`scripts/fetch_cq500.py`) | Reported as CC BY-NC-SA 4.0; confirm on headctstudy.qure.ai | Only a small fraction (see `data/cq500/screening.csv`): most scans are brain CTs ending at the skull base. |
| **NCI Imaging Data Commons** (Fedorov et al., *RadioGraphics* 2023), open TCIA mirrors | `pip install idc-index` (`scripts/fetch_idc.py`) | Per collection, mostly CC BY 4.0 (recorded per subject) | A few dozen head/whole-body CTs; many rejected for raised arms or cropped faces. Major head-and-neck collections (HNSCC, RADCURE, Head-Neck-PET-CT, OPC-Radiomics) are **not** in the open IDC buckets. |

**Screening result (2026-10-29, `scripts/fetch_cq500.py` and `scripts/fetch_idc.py`):**

| Source | Screened | Usable | Main reasons for rejection |
|---|---|---|---|
| CQ500 | 491 | 2 | 407 brain CTs that stop above the chin; 73 with only thick (> 2.5 mm) slices; 7 without a head; 1 arms/tissue beside head; 1 corrupt download |
| IDC open CT (head/neck/whole-body labels, ≥ 240 mm, ≤ 3 mm) | 78 | 1 | 77 contain no skull vault (mostly chest CTs labelled HEAD, or whole-body scans starting below the head) |

So the open sources give **3 usable subjects**: enough to verify the pipeline on real scans and to serve as a tiny public external test set, **not** for training or for the paper's main results.

## Needs an application (recommended for the paper)

| Source | What | How to get it |
|---|---|---|
| **New Mexico Decedent Image Database (NMDID)**, Univ. of New Mexico | > 15,000 decedents (2010–2017), whole-body post-mortem CT, 1 mm slices, ~60 metadata variables (age, sex, BMI, ancestry, cause of death) | Register with a university/research email at nmdid.unm.edu, then submit a research request (criteria, plan, expected results) that a committee approves; sign the data use agreement. Free. **Best fit for forensic work.** |
| **TCIA head-and-neck collections** (HNSCC, RADCURE, Head-Neck-PET-CT, OPC-Radiomics, HN1) | Thousands of radiotherapy/diagnostic head-and-neck CTs | Via cancerimagingarchive.net; several now require a data use agreement because faces are identifiable. Note: RT planning CT often has a thermoplastic mask pressing on the face (exclude or flag). |
| **HaN-Seg** (Podobnik et al., *Med Phys* 2023) | 42 head-and-neck CT + MR | Zenodo record 7442914 (4.9 GB). Check license and face coverage. |
| **Your own hospital data (Vietnam)** | Head CT with full face, ideally adults without facial trauma/surgery | Ethics committee approval, de-identification, data protection compliance. Valuable for a population-specific contribution. |

## Inclusion / exclusion criteria (apply to every source)

Automatic (`ffr/preprocess.py`): skull vault present, scan reaches ≥ 235 mm below the vertex, face not clipped by the field of view, no body tissue beside the head (raised arms), slice spacing ≤ 2.5–3 mm, no missing slices; gantry tilt is corrected from slice positions.

Manual (look at `scripts/qc_montage.py` output): exclude thermoplastic masks, bite blocks/open mouth, nasogastric or endotracheal tubes, heavy dental metal artefacts, facial fractures/surgery, severe decomposition (post-mortem), age < 18 unless studied separately.

Record every exclusion with its reason; the CONSORT-style flow diagram in the paper comes from `screening.csv`.
