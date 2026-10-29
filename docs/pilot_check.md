# Pipeline check on open data (not a result)

Date: 2026-10-29. Purpose: confirm that every stage runs on real CT, end to end. **These numbers must not be reported as results.**

- Subjects: the 3 open scans that passed QC (CQ500-CT-6, CQ500-CT-298, IDC cptac_sar C3N-00875).
- Splits: 3 folds, each with 1 training, 1 validation and 1 test subject (`scripts/make_splits.py --k 3`).
- U-Net: `configs/unet_pilot_cpu.yaml`, 40 epochs on 64³ patches, CPU, trained on **one** subject.
- Metrics: face ROI (front 80 mm, chin to brow), pooled over the 3 test subjects (mean ± SD). No identification test (gallery of 1 per fold).

| method            |   n | mean_mm     | rms_mm       | hd95_mm       | within_2mm   | signed_mean_mm   | lower_third_mm   | middle_third_mm   | upper_third_mm   |
|:------------------|----:|:------------|:-------------|:--------------|:-------------|:-----------------|:-----------------|:------------------|:-----------------|
| mean_face         |   3 | 6.13 ± 2.08 | 7.44 ± 2.45  | 15.42 ± 6.48  | 0.18 ± 0.10  | 0.24 ± 3.25      | 8.10 ± 3.49      | 5.69 ± 1.94       | 5.36 ± 2.09      |
| nearest_neighbour |   3 | 5.62 ± 2.64 | 6.68 ± 2.96  | 13.45 ± 5.75  | 0.21 ± 0.10  | -1.32 ± 3.08     | 7.64 ± 3.63      | 5.22 ± 2.32       | 4.54 ± 2.59      |
| ridge             |   3 | 5.52 ± 2.62 | 6.63 ± 3.11  | 13.78 ± 7.43  | 0.21 ± 0.11  | -0.80 ± 2.95     | 7.57 ± 3.74      | 5.18 ± 2.43       | 4.53 ± 2.75      |
| tissue_depth      |   3 | 7.95 ± 3.81 | 11.89 ± 4.85 | 33.93 ± 16.66 | 0.21 ± 0.09  | -5.50 ± 4.93     | 7.31 ± 2.95      | 6.12 ± 3.65       | 3.83 ± 0.32      |
| unet              |   3 | 6.72 ± 1.87 | 10.84 ± 2.07 | 33.98 ± 21.19 | 0.26 ± 0.12  | -0.83 ± 4.76     | 7.08 ± 1.53      | 6.50 ± 3.22       | 3.44 ± 0.98      |

What this shows: preprocessing, all baselines, training, sliding-window inference, meshing and metrics run on real scans and give errors in a plausible range (a few mm). What it does not show: anything about which method is better. With one training subject the U-Net cannot learn a skull→face mapping, and n = 3 has no statistical power. Real comparisons need the NMDID (or similar) cohort described in `experiment_plan.md`.
