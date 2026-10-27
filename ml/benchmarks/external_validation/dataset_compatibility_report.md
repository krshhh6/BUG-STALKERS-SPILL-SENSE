# SpillSense External Dataset Compatibility Report

## Summary

This report assesses the compatibility of five external datasets with the SpillSense production models (SpillSegNet segmenter and DualPolOilSpillNet classifier).

## Model Input Requirements

| Model | Architecture | Input Shape | Channels | Normalization | Threshold |
|-------|-------------|-------------|----------|---------------|-----------|
| SpillSegNet (segmenter) | Lightweight U-Net + SE | [1, 2, 512, 512] | VV (Ch0), VH (Ch1) | dB clip-normalize: VV [-32, -10], VH [-42, -20] | 0.50 |
| DualPolOilSpillNet (classifier) | DS-Conv + SE | [1, 2, 400, 400] | VV (Ch0), VH (Ch1) | Same dB normalization | 0.49 (production) / 0.43 (staged) |

Both models require **dual-polarization** (VV + VH) Sentinel-1 SAR data in **Sigma0 dB** representation.

---

## Dataset A: Refined Deep-SAR Oil Spill (SOS)

| Attribute | Value |
|-----------|-------|
| Source | [Zenodo 15298010](https://zenodo.org/records/15298010) |
| License | CC BY 4.0 (inferred) |
| Format | ZIP archives (images.zip ~1.1 GB, masks.zip ~27 MB) |
| Polarization | **Single-polarization** (VV intensity or grayscale-rendered) |
| Image format | PNG/JPG patches (not raw GeoTIFF) |
| Mask format | Binary PNG (oil=foreground, background=0) |
| Resolution | Variable patch sizes |
| Calibration | Unknown (likely not raw Sigma0 dB) |

### Compatibility Assessment

- **INCOMPATIBLE for direct zero-shot evaluation** — dataset is single-polarization while our models require dual-pol (VV + VH).
- **Adapted evaluation possible**: grayscale channel duplicated as VV, with synthetic VH offset (-10.5 dB physical approximation). This is **not** scientifically equivalent to genuine dual-pol data.
- Images are pre-rendered (not raw GeoTIFF), so the dB normalization pipeline cannot be applied directly.
- Evaluation clearly labelled as **"adapted single-pol-to-dual-pol"**, not zero-shot.

### Published Baseline Comparison

The SOS dataset is associated with a DeepLabV3-based hard-negative approach (Zhu et al., 2021, IEEE TGRS). Published results can be compared contextually but not directly, since evaluation protocols differ.

---

## Dataset B: DARTIS (PANGAEA 980773)

| Attribute | Value |
|-----------|-------|
| Source | [PANGAEA 980773](https://doi.pangaea.de/10.1594/PANGAEA.980773) |
| License | CC BY 4.0 |
| Format | Tabular metadata (TSV) with coordinates, not image patches |
| Content | 1365 oil patches + 2290 no-oil patches (metadata only) |
| Polarization | Sentinel-1 (VV+VH available in source scenes) |

### Compatibility Assessment

- **BLOCKED** — DARTIS provides annotation metadata and coordinates, not pre-cropped image patches.
- Reproducing the dataset requires downloading full Sentinel-1 scenes from the Copernicus Open Access Hub using the provided scene IDs and coordinates, then cropping patches.
- This requires Copernicus API credentials and significant download time (full S1 scenes are ~1-7 GB each).
- If the full scenes were obtained, the dataset would be **compatible** with our dual-pol models.

---

## Dataset C: CSIRO Sentinel-1 SAR Data

| Attribute | Value |
|-----------|-------|
| Source | [data.csiro.au](https://data.csiro.au/) |
| License | Unknown |
| Format | Unknown |
| Content | No specific oil spill benchmark identified |

### Compatibility Assessment

- **BLOCKED** — no specific CSIRO SAR oil spill benchmark dataset with pixel-level segmentation masks was identified on the CSIRO Data Access Portal.
- The portal hosts many geospatial datasets but none appeared to match the requirements of a standardized oil spill detection benchmark.
- Further manual exploration of the portal may identify relevant collections.

---

## Dataset D: Cantarell Oil Spill/Seepage Segmentation Dataset

| Attribute | Value |
|-----------|-------|
| Source | [GitHub Garcia-INPE](https://github.com/Garcia-INPE/oilspill-detection-dataset) |
| License | CC BY 4.0 |
| Format | 173 image-mask triplets in PNG format |
| Image size | ~800 x 600 pixels |
| Polarization | **Single-channel** (RGB-rendered SAR, not raw dual-pol) |
| Mask format | Binary: 0=sea, 1=slick |
| Region | Gulf of Mexico, Cantarell complex |
| Period | Jan-Sep 2020 |
| Splits | 108 train / 43 val / 22 test |

### Compatibility Assessment

- **INCOMPATIBLE for direct zero-shot evaluation** — images are RGB-rendered SAR (3-channel), not raw dual-pol GeoTIFF.
- **Adapted evaluation possible**: convert RGB to grayscale, treat as VV, synthesize VH with offset.
- The 800x600 pixel dimensions require resizing to 512x512 for the segmenter.
- Evaluation clearly labelled as **"adapted RGB-SAR-to-dual-pol"**.

### Published Baselines

The dataset includes three baselines:
1. **Felzenszwalb** (unsupervised classical segmentation)
2. **TinyUNet** (deep learning)
3. **Hybrid** (Felzenszwalb + Random Forest)

Results from `pipeline_comparison.csv` can be directly compared.

---

## Dataset E: Sentinel-1 SAR Oil Spill Dataset (Trujillo-Acatitla et al.)

| Attribute | Value |
|-----------|-------|
| Source | [Zenodo 8346860](https://zenodo.org/records/8346860) |
| License | CC BY 4.0 |
| Format | 2048x2048x2 GeoTIFF (Sigma0 dB) |
| Polarization | **Dual (VV + VH)** — fully compatible |
| Masks | Binary GeoTIFF (fg=1, bg=0) |

### Compatibility Assessment

- **THIS IS OUR TRAINING DATASET** — Parts I, II, and III of this Zenodo collection were used to train and validate the SpillSense production models.
- Using this dataset for "external validation" would be **scientifically invalid**.
- Any evaluation on this dataset is labelled as **internal validation**, not external.
- The Part III test set evaluation was already performed and reported as our reference metrics.

---

## Summary Matrix

| Dataset | Compatible | Polarization | Evaluation Type | Status |
|---------|-----------|-------------|-----------------|--------|
| A: SOS | Adapted | Single-pol | Adapted (synth VH) | Downloadable |
| B: DARTIS | Blocked | Dual-pol (source) | N/A | Requires Copernicus API |
| C: CSIRO | Blocked | Unknown | N/A | No dataset identified |
| D: Cantarell | Adapted | RGB SAR | Adapted (synth VH) | Downloadable |
| E: Zenodo S-1 | Training data | Dual-pol | INVALID as external | Excluded |

> **Critical Finding**: No publicly available external dataset provides pre-processed dual-polarization (VV + VH) Sentinel-1 SAR patches with segmentation masks in a format directly compatible with our production models. All feasible evaluations require channel synthesis adaptation, which must be clearly disclosed.
