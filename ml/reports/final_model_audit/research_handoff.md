# SpillSense (SIH-26143) — IEEE Research Paper Handoff Guide

**Target Task:** IEEE Research Paper Critical Audit & Manuscript Alignment  
**Timestamp:** October 14, 2026  
**Auditor:** Team BUG STALKERS / Antigravity AI Pair  

---

## 1. Executive Purpose
This guide establishes the boundary between verified empirical facts and unproven claims. When auditing the IEEE paper manuscript, the research team must enforce these strict guidelines to ensure 100% scientific defensibility.

---

## 2. Definitive Verified Numbers (Safe to Cite)

### 2.1 Full-Scene Segmentation Performance (Phase 4 Independent Validation)
- **Evaluation Split:** 20 strictly held-out Sentinel-1 scenes (`val_scenes[21:41]`, $2048 \times 2048 \times 2$, 83,886,080 pixels total).
- **Candidate Model:** SpillSegNet (`best_segmenter_v3_exp1_lr1e4.pt` / `oil_segmenter.onnx`, SHA-256: `D464E6CEF8DC47B7AFDD0E7F0AE1068AE92798DF3154004BB771B7ED317D4A0E`).
- **Baseline Model:** `best_segmenter_v3.pt` / `oil_segmenter_baseline_backup.onnx` (SHA-256: `8E58A9AF1FB3D3F0C94A79DAAA17DA2015910F82D2A74EBCFE02B0CC60BC3DE9`).
- **Operating Threshold:** $\tau = 0.50$, Tile Size $512 \times 512$, Stride 384 (25 overlapping tiles).
- **Verified Micro (Pixel-Pooled) Metrics:**
  - Candidate Micro Dice: **0.8390** vs. Baseline **0.5945** (+0.2445 improvement).
  - Candidate Micro IoU: **0.7227** vs. Baseline **0.4230** (+0.2997 improvement).
  - Candidate Micro Precision: **0.8013** vs. Baseline **0.4509** (+0.3504 improvement).
  - Candidate Micro Recall: **0.8804** vs. Baseline **0.8722** (+0.0082 improvement).
  - False Positive Pixels: Reduced by **79.4%** (Candidate: **297,985** vs. Baseline: **1,449,897**).
- **Verified Macro (Per-Scene Mean) Metrics:**
  - Candidate Macro Dice: **0.8219** (std: 0.1598) vs. Baseline **0.7556** (std: 0.1981).
  - Candidate Macro IoU: **0.7199** (std: 0.1748) vs. Baseline **0.6355** (std: 0.1906).
  - Candidate Macro Precision: **0.8255** vs. Baseline **0.7235**.

### 2.2 Curated Slick-Patch Segmentation Performance
- **Evaluation Split:** 62 curated foreground crops from 21 scenes (`val_scenes[:21]`, 16.25M pixels total).
- **Baseline:** Dice **0.8279**, IoU **0.7063**, Precision **0.7987**, Recall **0.8593**.
- **Candidate:** Dice **0.8750**, IoU **0.7778**, Precision **0.8891**, Recall **0.8613**.
- *Paper Constraint:* Must clearly state that this metric evaluates cropped slick regions only, where open-ocean background is excluded.

### 2.3 Hardware Latency & Inference Throughput
- **Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU, Intel Core i7, 16 GB RAM.
- **Full Ingest-to-Attribution Pipeline:**
  - Mean Latency: **1483.93 ms** (~1.48 seconds per full $2048 \times 2048$ scene).
  - Median Latency: **1413.86 ms**.
  - 95th Percentile Latency: **2420.14 ms**.
- **Classifier Latency:**
  - PyTorch CUDA (RTX 4060): **9.56 ms** median.
  - ONNX Runtime C++ CPU: **5.34 ms** median.
  - In-Browser WebAssembly: **12.38 ms** median.
- **Segmenter Tiled Latency:**
  - Stride 384 (25 tiles, 25% overlap): **472.63 ms** pure GPU inference.
  - End-to-end full scene tiled processing (I/O + transfers + blending): **1172.30 ms** median.

### 2.4 Polarization & Physical Modeling
- Dual-polarization is essential:
  - Disabling VH (VV-only) causes oil detection recall to drop from **1.000 to 0.000**.
  - Disabling VV (VH-only) causes ocean surface TNR to drop from **0.900 to 0.000**.
- Radiometric calibration bound: Clamped to $[-32.0\text{ dB}, -10.0\text{ dB}]$, matching ESA Sentinel-1 IW GRD calibrated ocean backscatter limits.

---

## 3. Corrected Numbers & Erroneous Claims to Reject

| Previous Inaccurate / Contaminated Claim | Verified Empirical Fact | Reason for Correction |
|---|---|---|
| "Baseline segmenter scores 0.6811 on held-out test data" | **Baseline scores 0.5945 Micro Dice on held-out test scenes** | The 0.6811 figure was Macro Dice on a 50-scene benchmark that included training scenes (0..97). On genuine held-out scenes, baseline Micro Dice is 0.5945. |
| "50-scene full pipeline took 107.15s of GPU processing" | **50 scenes took 74.20s of GPU processing; 107.15s was the whole script duration** | The script executed 3 full-scene CPU runs (25.2s), CPU patch runs (15.75s), and WASM runs in addition to the 50 GPU scenes. |
| "Boundary IoU peaks at threshold 0.50" | **Boundary IoU increases monotonically up to threshold 0.80 (0.2140)** | Threshold 0.50 yields Boundary IoU 0.1222. Boundary IoU rises as threshold tightens to the high-confidence slick core. |
| "Classifier achieved 0.9767 F1 on independent test set" | **0.9767 F1 was achieved on the validation split (`cls_val_v3_zenodo_gt_dualpol.npz`)** | The independent Zenodo Part III test set download is in progress (13.17%). |
| "Candidate classifier achieved 2.67% ship-wake FPR on independent test set" | **Ship-wake FPR is UNMEASURABLE on Zenodo Part III** | Zenodo Part III contains zero ship-wake annotations. The 2.67% FPR was measured on 75 curated validation wake crops. |
| "OpenDrift 4D Lagrangian drift reversal quantitatively verified" | **OpenDrift library is NOT INSTALLED; drift simulation was UNEXECUTED** | `latency_results.json` explicitly logs drift stage as `count: 0, status: "Unavailable"`. Must be cited as future work or conceptual architecture. |

---

## 4. Checklist for the IEEE Paper Audit

When reviewing each section of the IEEE manuscript:

- [ ] **Abstract:** Verify that reported segmentation Dice is explicitly identified as either Micro Dice (0.8390) on full scenes or Crop Dice (0.8750). Never conflate them.
- [ ] **Abstract:** Verify that pipeline throughput is stated as **~1.48 seconds per scene** on an edge GPU (RTX 4060), and avoid citing the 107.15s test suite duration as operational throughput.
- [ ] **Methodology (Tiling):** Confirm that the tiling grid is stated as **$5 \times 5 = 25\text{ tiles}$** with stride 384 (25% overlap) for a $2048 \times 2048$ image, yielding a smooth seam error ratio of 0.922x.
- [ ] **Methodology (Radiometric Bounds):** Confirm that Sentinel-1 IW GRD input normalization uses $[-32.0\text{ dB}, -10.0\text{ dB}]$.
- [ ] **Results (Segmentation Table):** Include both Micro Dice (0.8390 vs. 0.5945) and Macro Dice (0.8219 vs. 0.7556) with clear mathematical definitions of pooling vs. per-scene averaging. Highlight the **79.4% reduction in false positive pixels** (1,449,897 down to 297,985).
- [ ] **Results (Classification Table):** Accurately label the 0.9767 F1 and 2.67% ship-wake FPR as evaluated on the **held-out validation cache**, not the independent Zenodo Part III test set.
- [ ] **Ablation Studies:** Cite the polarization ablation accurately: show that VV-only removes oil recall (0.0) while VH-only removes ocean background TNR (0.0).
- [ ] **Attribution & Drift Section:** Emphasize the executed 5-factor kinematic AIS attribution engine (0.16 ms latency) and present OpenDrift Lagrangian trajectory reversal as a modular conceptual integration.
