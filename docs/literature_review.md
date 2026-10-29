# Literature review: skull-to-face (craniofacial) reconstruction

Scope: computerized forensic facial reconstruction (FFR / craniofacial reconstruction, CFR), i.e. predicting the facial soft tissue surface from a skull, with a focus on data-driven and deep-learning methods and on how results are evaluated.

> Status markers. **[read]** = details checked against the paper or its abstract page.
> **[verify]** = found and relevant, but full text not read yet; check numbers and author lists before citing.

## 1. Background and classical methods

| Work | Idea | Notes for us |
|---|---|---|
| Wilkinson, *Forensic Facial Reconstruction*, Cambridge Univ. Press, 2004 | Manual/anatomical (Manchester) method | Standard reference for the forensic context and for what "recognition" means in practice. |
| Stephan & Simpson, *J Forensic Sci* 2008, "Facial soft tissue depths in craniofacial identification (part I)" | Meta-analysis of soft tissue depth (FSTD) at landmarks | Source of population tissue-depth tables; our *constant tissue depth* baseline is the simplest form of this idea. |
| Claes et al., *Forensic Sci Int* 201 (2010) 138–145, "Computerized craniofacial reconstruction: conceptual framework and review" | Review; statistical (PCA) skull–face models | Defines the template-deformation vs. statistical-model families. Our *ridge* baseline is a linear statistical skull→face model in the same family. |
| Tilotta et al., *Forensic Sci Int* 191 (2009) 112.e1–112.e12, "Construction and analysis of a head CT-scan database for craniofacial reconstruction" | Head-CT database (≈ 85 French subjects) + analysis | Shows CT is the standard data source; dataset is not openly released. [verify] |
| Gietzen et al., *PLoS ONE* 2019 (arXiv 1808.07334), "A method for automatic forensic facial reconstruction based on dense statistics of soft tissue thickness" | Statistical skull model + dense FSTT + PCA head model | 60 head CTs (Mainz) + 39 surface scans; evaluation of the face estimate was qualitative on a single case. [read] |

## 2. Deep-learning methods

| Work | Representation | Data | Reported evaluation |
|---|---|---|---|
| *Security and Communication Networks* 2021, doi 10.1155/2021/9987792 (CFPRN) | Skull and face projected to 2D cylindrical elevation maps; U-shaped CNN; MSE | 1,447 cone-beam head CTs (Nanjing, not public); 1,310 train / 137 val | PSNR/SSIM; text states error "within 2 mm" for most of the face (and "1 mm" elsewhere, inconsistent). [read] |
| Zhang et al., ACM Multimedia 2022, doi 10.1145/3503161.3548254 | Depth maps + conditional GAN (CGAN-CFG) | private CT | [verify] |
| CR-GAN, *Pattern Recognition* (ScienceDirect PII S0031320321005768), "Automatic craniofacial reconstruction for personal identification" | GAN on 2D projections; evaluates identification | private CT | Important: explicitly targets *personal identification*. [verify] |
| Liang et al., arXiv 2403.16207 (2024), "Skull-to-Face: Anatomy-Guided 3D Facial Reconstruction and Editing" | SDXL portrait → DECA/FLAME 3D face → fit to tissue-depth landmarks | Skull100 (50/50 split), from whole-head CT | Normalised mean error ≈ 1.67–1.76 (%); qualitative comparison to Pix2Pix, CycleGAN, HF-GGR, SCULPTOR. [read] |
| **Int J Legal Med 2026**, doi 10.1007/s00414-026-04016-4, "Deep learning-based 3D forensic facial reconstruction: a volumetric morphological inference approach using a skull-conditioned U-Net" | Volumetric, skull-conditioned U-Net | ? | **Closest prior work to this repository. Read in full before writing the paper and position against it.** [verify] |
| Int J Legal Med 2026, doi 10.1007/s00414-026-04000-y | AI-based modelling of craniofacial soft tissue thickness from CT, NW Indian population | CT | Population-specific FSTT. [verify] |
| Cranio-Diff, arXiv 2606.09699 (2026) | Diffusion, cross-domain reconstruction from 2D X-ray skull | — | Generative alternative. [verify] |

Related volumetric work on skulls (not faces), useful for architecture and data precedent:
- 3D U-Net (Çiçek et al., MICCAI 2016) — the base architecture.
- Cranial implant design / skull completion with 3D U-Nets on CQ500-derived skulls (SkullFix/SkullBreak; AutoImplant challenge, Li & Egger 2020–2021; Wu et al., *Front Bioeng Biotechnol* 2023, 10.3389/fbioe.2023.1297933). Shows CQ500 is accepted as a skull source, but those works only need the cranium, not the face. [read]

## 3. What the field reports, and the gaps

1. **Metrics.** mm surface distance (mean / RMS / percentiles, colour error maps), regional errors (eyes, nose, mouth are worst), and sometimes PSNR/SSIM on 2D projections. Identification tests (does the reconstruction pick the right person from a gallery?) are rare but are what matters for forensic use.
2. **Data.** Almost every deep model is trained on a private hospital dataset of a single population. No open benchmark exists, so results are not comparable across papers.
3. **Representation.** Most deep methods work on 2D projections (elevation/depth maps). Volumetric SDF prediction is newer (see IJLM 2026).
4. **Uncertainty** is almost never reported, although a reconstruction is a probabilistic estimate.

## 4. Where this project can be novel (pick 2–3 for the paper)

Because a skull-conditioned U-Net already exists (IJLM 2026), "we use a 3D U-Net" is not a contribution by itself. Defensible contributions:

- **A: Open, reproducible benchmark.** Leakage-free, skull-only anchoring; documented QC; public-data subset (CQ500/IDC full-face scans) with fixed splits and baselines anyone can rerun.
- **B: Identification-centred evaluation.** Closed-set gallery retrieval (rank-1/5, CMC) next to mm error, with gallery-size analysis.
- **C: Uncertainty maps** (deep ensembles or MC dropout) and calibration of predicted error vs. actual error.
- **D: Post-mortem vs. clinical CT domain shift** (NMDID PMCT vs. clinical CT), the real forensic setting.
- **E: Biological-profile conditioning** (sex, age, BMI/ancestry via FiLM) and fairness across groups.
- **F: Population-specific model** (e.g. Vietnamese subjects) with cross-population testing.
