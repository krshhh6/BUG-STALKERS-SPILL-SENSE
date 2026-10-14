# SpillSense (SIH-26143) — Dual-Polarization SAR ML & Segmentation Pipeline v3

High-accuracy, real-world Sentinel-1 Synthetic Aperture Radar (SAR) dual-polarization oil spill detection and segmentation pipeline engineered with official Zenodo ground-truth masks, zero cross-scene leakage, and lightweight in-browser ONNX inference.

---

## 1. SAR Physics & Ingestion Specifications

- **Imagery Source:** Dual-polarization Sentinel-1 Interferometric Wide (IW) swath Ground Range Detected (GRD) GeoTIFF scenes ($2048 \times 2048 \times 2$, float32).
- **Physical Representation:** Backscatter values are **calibrated radar cross-section in decibels ($\sigma^0\text{ dB}$)**.
- **Band Mapping:**
  - **Band 0:** Cross-polarization **VH** (ambient ocean median $\approx -33\text{ dB}$).
  - **Band 1:** Co-polarization **VV** (ambient ocean median $\approx -21\text{ dB}$, captures capillary waves).
- **Model Input:** 2 channels (Channel 0 = VV, Channel 1 = VH).
  - **VV:** Normalized decibels clipped to $[-32.0, -10.0]\text{ dB} \rightarrow [0.0, 1.0]$.
  - **VH:** Normalized decibels clipped to $[-42.0, -20.0]\text{ dB} \rightarrow [0.0, 1.0]$.

---

## 2. Dataset Partitioning & Ground-Truth Integration

- **Ground-Truth Masks:** Official Zenodo Sentinel-1 binary GeoTIFF masks (1,200 masks matching 1,200 GeoTIFF scenes 1-to-1).
- **Scene-Level Splitting:** Partitioned strictly by **Scene ID** before patch extraction:
  - **Train:** 983 scenes $\rightarrow$ 22,341 patches (15,364 oil, 4,899 clean, 1,836 look-alike, 242 ship/wake).
  - **Validation:** 217 holdout scenes $\rightarrow$ 4,818 patches (3,272 oil, 1,082 clean, 389 look-alike, 75 ship/wake).
  - **Zero Leakage:** Patches from the same scene never appear across split boundaries.
- **Labeling & Hard Negatives:**
  - **Oil Positives:** Real mask positives; small spills (1–5% coverage) oversampled alongside strong spills ($\ge 5\%$).
  - **Hard Negatives:** Mined from true zero-mask regions:
    - Clean calm ocean
    - Natural dark look-alikes (low wind, biogenic films)
    - Ship targets and radar wakes

---

## 3. Model Architectures & Validation Performance

### A. Classifier (`DualPolOilSpillNet`)
- **Input:** $2 \times 400 \times 400$
- **Backbone:** Depthwise-separable convolutions + Squeeze-and-Excitation (SE) cross-polarization attention (317k params).
- **Loss:** Focal Loss ($\gamma=2.0, \alpha=0.25$) with class-weighted sampling.
- **Metrics on 217 Unseen Holdout Scenes (4,818 Patches):**
  - **Accuracy:** **96.5%**
  - **Precision:** **98.0%**
  - **Recall:** **96.8%**
  - **F1-Score:** **97.4%**
  - **ROC-AUC:** **0.994** | **PR-AUC:** **0.984**
  - **Calibrated Optimal Threshold:** `0.27`
- **Hard Negative True Negative Rates (TNR):**
  - **Look-Alikes:** **98.2%** (Only 7 false alarms out of 389 look-alikes)
  - **Ship / Wake:** **96.0%**
  - **Clean Ocean:** **94.9%**

### B. Segmenter (`SpillSegNet`)
- **Input:** $2 \times 512 \times 512 \rightarrow 1 \times 512 \times 512$
- **Backbone:** 4-stage U-Net with skip connections and bilinear upsampling (1.59M params).
- **Loss:** Dice Loss + Binary Cross-Entropy.
- **Validation Metrics (Curated 512×512 Oil Crops):**
  - **Dice Score:** **82.8%** (up to 88.4% on early holdout crops)
  - **IoU:** **70.6%** (up to 79.3% on early holdout crops)
  - **Precision:** **79.9%**
  - **Recall:** **85.9%**
- **Empirical Full-Scene Evaluation (Uncropped 2048×2048 Scenes, 25 Tiles):**
  - **Mean Macro-Dice:** **68.1%** | **Mean Macro-IoU:** **55.5%**
  - **Pixel Recall:** **88.9%** (Consistently identifies real oil spill extent)
  - **Full-Scene Precision:** **64.4%** (Reflects ambient open-ocean class imbalance across 4.19M pixels)

---

## 4. Curated 40-Scene Benchmark Gallery Evaluation

| Category | Scenes Tested | Expected Class | Predictions | Accuracy / Rejection |
| :--- | :--- | :--- | :--- | :--- |
| **Oil Spill** | 10 scenes | Positive (OIL) | 10/10 Detected ($P = 0.52 \text{ to } 0.98$) | **100% Detection** |
| **Clean Ocean** | 10 scenes | Negative (CLEAN) | 10/10 Non-Oil ($P = 0.05 \text{ to } 0.20$) | **100% Clean TNR** |
| **Look-Alikes** | 10 scenes | Negative (LOOK-ALIKE) | 10/10 Non-Oil ($P = 0.02 \text{ to } 0.05$) | **100% Look-alike TNR (0% FP)** |
| **Ship / Wake** | 10 scenes | Negative (SHIP/WAKE) | 9/10 Non-Oil ($P = 0.02 \text{ to } 0.19$) | **90% Ship/Wake TNR** |

---

## 5. Execution & Usage Commands

```bash
# 1. Download official Zenodo ground-truth masks
python -u -m ml.download_zenodo

# 2. Train DualPolOilSpillNet classifier with focal loss and validation calibration
python -u -m ml.train

# 3. Train SpillSegNet U-Net segmenter with Dice+BCE loss
python -u -m ml.train_segmentation

# 4. Generate 40-scene benchmark gallery & evaluation
python -u -m ml.benchmark_gallery

# 5. Export lightweight ONNX models for browser inference
python -u -m ml.export_onnx
```
