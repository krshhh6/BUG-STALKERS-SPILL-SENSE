# SpillSense SIH-26143 — Phase 3 Validation Audit Report

**Date:** 2026-10-14  
**Audit Author:** Antigravity Pair-Programming Assistant  
**Git Branch:** `improvement/phase0-backup`  
**Evaluation Scope:** Baseline vs. Candidate Checkpoints (Classifier & Segmenter)  
**Status:** COMPLETE — Candidate Models Frozen & Audited

---

## 1. Checkpoint Verification and Configuration Freeze

All original and Phase 3 candidate checkpoint files have been verified using SHA-256 cryptographic hashes. The baseline checkpoints remain bit-for-bit identical to the Phase 0 backup record.

### Checkpoint SHA-256 Registry

| Checkpoint Name | Model Architecture | Role | SHA-256 Hash | Status |
|---|---|---|---|---|
| `best_classifier_v3.pt` | DualPolOilSpillNet | Original Baseline (Gallery) | `076A5AA4224FA90050EB29526CD9AD8772D4CF38257FCCF56B43AD09A4930705` | Verified Original |
| `best_classifier_v3_finetuned.pt` | DualPolOilSpillNet | Finetuned Baseline | `DA224DE8D0C41CC39D7E1C8606658D16AB786334A9307D96715E524D16A88F6D` | Verified Original |
| `best_dualpol_classifier.pt` | DualPolOilSpillNet | Original Baseline (Held-out) | `262EF1A8D1F3FBCFBBEDBCC67AEE0C3C1FAF5995CC855CF7131D1FD14C78D902` | Verified Original |
| `best_segmenter_v3.pt` | SpillSegNet (U-Net) | Original Baseline | `9C5901016C19CB93B8B492512936D8726E1B988F32AF8E392A5225C260C08B08` | Verified Original |
| `best_dualpol_classifier_exp1_loss_alpha05.pt` | DualPolOilSpillNet | Candidate Exp 1A (Loss) | `5531E2B73537E20A6A00404B6D33B699B428A18144B975AA606EDCB363A5FE31` | Frozen Candidate |
| `best_dualpol_classifier_exp2_sampling_sw30.pt` | DualPolOilSpillNet | Candidate Exp 1B (Sampling) | `014D209A19DBD5C42DA275648F8E5BCB748ABCFCB1562B91F0541FBDCCE0653E` | Frozen Candidate |
| `best_segmenter_v3_exp1_lr1e4.pt` | SpillSegNet (U-Net) | Candidate Exp 1 (LR/Patience)| `38DD8341C2CD0D1DD0DDB5082130C96B675CD991F8FB692E7B206D1831B3584D` | Frozen Candidate |

### Checkpoint Configurations

```json
{
  "best_dualpol_classifier.pt": {
    "epoch": 7,
    "best_threshold": 0.49,
    "loss": "focal (alpha=0.25, gamma=2.0)",
    "sampling_weights": {"oil": 0.40, "lookalike": 0.30, "clean": 0.20, "ship_wake": 0.10},
    "lr": 0.001,
    "patience": 7
  },
  "best_dualpol_classifier_exp1_loss_alpha05.pt": {
    "epoch": 3,
    "best_threshold": 0.47,
    "loss": "focal (alpha=0.50, gamma=2.0)",
    "sampling_weights": {"oil": 0.40, "lookalike": 0.30, "clean": 0.20, "ship_wake": 0.10},
    "lr": 0.001,
    "batch_size": 64,
    "patience": 3,
    "seed": 42
  },
  "best_dualpol_classifier_exp2_sampling_sw30.pt": {
    "epoch": 4,
    "best_threshold": 0.43,
    "loss": "focal (alpha=0.25, gamma=2.0)",
    "sampling_weights": {"oil": 0.30, "lookalike": 0.25, "clean": 0.15, "ship_wake": 0.30},
    "lr": 0.001,
    "batch_size": 64,
    "patience": 3,
    "seed": 42
  },
  "best_segmenter_v3.pt": {
    "epoch": 2,
    "loss": "0.6*Dice + 0.4*BCE",
    "lr": 0.0005,
    "patience": 6,
    "max_scenes": 120,
    "seed": 42
  },
  "best_segmenter_v3_exp1_lr1e4.pt": {
    "epoch": 29,
    "loss": "0.6*Dice + 0.4*BCE",
    "lr": 0.0001,
    "patience": 12,
    "max_scenes": 120,
    "seed": 42
  }
}
```

---

## 2. Dataset Separation & Data Integrity Audit

### 2.1 Scene Partitioning Audit
The dataset was audited across its scene-level identifiers to verify separation between training and validation:

1. **Classifier Splits (`manifest_v3_zenodo_gt_dualpol.json`):**
   - Training Scenes: **983 scenes**
   - Validation Scenes: **217 scenes**
   - Train $\cap$ Validation Scene Overlap: **0 scenes** (Strict scene-level partitioning enforced)

2. **Segmenter Splits (`ml/train_segmentation.py`):**
   - Training Scenes: **98 scenes** (`train_scenes[:98]`)
   - Validation Scenes: **21 scenes** (`val_scenes[:21]`, yielding 62 validation crops)
   - Train $\cap$ Validation Scene Overlap: **0 scenes**

3. **Historical Split Discovery (V2 vs. V3):**
   - The original baseline `best_dualpol_classifier.pt` was trained on `v2_dualpol_400px_scenesplit_v1` (840 train scenes, 180 validation scenes).
   - In V3, the 1,200 scenes were reshuffled with an 82/18 train/val split.
   - Crucially, all 180 scenes designated as "test" in V2 were allocated to V3 Validation. Zero V2 test scenes were included in V3 Train.

### 2.2 Duplicate and Near-Duplicate Crop Analysis

A byte-level MD5 collision audit was conducted across all 22,341 training patches and 4,818 validation patches in `cls_train_v3` and `cls_val_v3`:

- **Total Unique Patch Hashes:**
  - Training: 15,707 unique hashes across 22,341 patches (reflects oversampled small-spill crops)
  - Validation: 3,449 unique hashes across 4,818 patches
- **Cross-Split MD5 Collisions:** Exactly **27 hashes** were identical between training and validation.

#### Root-Cause Investigation of the 27 Collisions:
1. **Constant Background Patch (`Hash e165a0b9`):**
   - Value: Uniform saturated border return (`min=255, max=255, std=0.0`).
   - Appears in 26 training scenes and 16 validation scenes due to identical nodata ocean edge clipping.
2. **Zenodo Source Dataset Scene Duplicates:**
   - Exhaustive MD5 hashing of all 1,200 original TIFF files revealed that the Zenodo Part I archive contains exactly **two identical duplicate scene pairs**:
     - `00356.tif` $\equiv$ `00357.tif` (100% bit-for-bit identical, mean absolute difference = 0.0)
     - `00007.tif` $\equiv$ `01339.tif` (100% bit-for-bit identical, mean absolute difference = 0.0)
   - During random scene shuffling, `00356` was assigned to train and `00357` to validation; `00007` to train and `01339` to validation.
   - This caused 26 non-trivial patches to be shared between splits (representing only **0.54%** of the validation set).

#### Deduplicated Re-Evaluation:
Excluding all duplicate hashes leaves a strictly clean validation set of **4,734 patches**. Evaluating models on this clean set yielded identical performance:
- Baseline: Accuracy 0.5171, Recall 0.3149, F1 0.4716, Ship-wake FPR 0.3733
- Exp 1A (Loss): Accuracy 0.9622, Recall 0.9574, F1 0.9719, Ship-wake FPR 0.0933
- Exp 1B (Sampling): Accuracy 0.9683, Recall 0.9620, F1 0.9765, Ship-wake FPR 0.0267

**Conclusion:** The 27 duplicate patches had no statistically meaningful impact on validation metrics. Data leakage is ruled out as an explanation for candidate gains.

---

## 3. Threshold Selection Methodology Audit

### Mechanism
In `ml/train.py`, `find_optimal_threshold` sweeps 46 candidate thresholds from 0.05 to 0.95 on validation predictions at the end of each epoch, selecting the threshold that maximizes:
$$\text{Score} = 0.50 \cdot \text{F1} + 0.30 \cdot \text{TNR}_{\text{lookalike}} + 0.20 \cdot \text{PR-AUC}$$

### Assessment of Selection Bias
Because the threshold (0.47 for Exp 1A, 0.43 for Exp 1B) was tuned on the validation set, the reported validation metrics could theoretically reflect threshold overfitting.

To audit this bias, all models were evaluated at a **fixed, untuned standard threshold of 0.50**:

| Model | Tuned Threshold | Tuned F1 | Tuned Recall | Fixed Thresh (0.50) F1 | Fixed Thresh (0.50) Recall | Delta from Tuning |
|---|---|---|---|---|---|---|
| **Baseline** | 0.49 | 0.4712 | 0.3145 | 0.4683 | 0.3117 | -0.0029 |
| **Exp 1A ($\alpha=0.50$)** | 0.47 | 0.9722 | 0.9578 | 0.9696 | 0.9514 | -0.0026 |
| **Exp 1B ($w_{\text{wake}}=0.30$)** | 0.43 | 0.9767 | 0.9624 | 0.9657 | 0.9386 | -0.0110 |

**Finding:** Threshold tuning contributes only $\approx 0.003$ to $0.011$ in F1 score. The massive improvement from F1=0.47 to F1=0.97 exists robustly at the standard threshold 0.50 without any threshold tuning.

---

## 4. Reproducible Baseline-versus-Candidate Evaluation

Evaluated on the identical held-out validation set ([`cls_val_v3_zenodo_gt_dualpol.npz`](file:///d:/Spill%20Sense/SIH-26143-OIL-Spill/ml/cache_patches/cls_val_v3_zenodo_gt_dualpol.npz), 4,818 patches) and segmenter validation split (21 scenes, 62 crops, 16.25M pixels):

### 4.1 Classifier Comparative Metrics Table

| Metric | Baseline (`best_dualpol_classifier.pt`) | Exp 1A: Loss ($\alpha=0.50$) | Exp 1B: Sampling ($w_{\text{wake}}=0.30$) |
|---|---|---|---|
| **Best Epoch** | 7 | 3 | 4 |
| **Threshold** | 0.49 | 0.47 | 0.43 |
| **Accuracy** | 52.05% | 96.28% | **96.89%** |
| **Precision** | 0.9389 | 0.9871 | **0.9915** |
| **Recall (Oil)** | 31.45% (1,029 / 3,272) | 95.78% (3,134 / 3,272) | **96.24% (3,149 / 3,272)** |
| **F1 Score** | 0.4712 | 0.9722 | **0.9767** |
| **ROC-AUC** | 0.6985 | 0.9949 | **0.9959** |
| **PR-AUC** | 0.5778 | 0.9500 | **0.9736** |
| **Ship-Wake FPR** | **37.33% (28 / 75 FPs)** | 9.33% (7 / 75 FPs) | **2.67% (2 / 75 FPs)** |
| **Ship-Wake TNR** | 62.67% (47 / 75 TNs) | 90.67% (68 / 75 TNs) | **97.33% (73 / 75 TNs)** |
| **Look-alike FPR** | 6.94% (27 / 389 FPs) | **1.03% (4 / 389 FPs)** | 1.80% (7 / 389 FPs) |
| **Clean FPR** | 1.11% (12 / 1,082 FPs) | 2.77% (30 / 1,082 FPs) | 1.66% (18 / 1,082 FPs) |
| **Total False Positives** | 67 | 41 | **27** |
| **Total False Negatives** | 2,243 | 138 | **123** |

---

### 4.2 Segmenter Curated-Patch Comparative Metrics Table

| Metric | Baseline (`best_segmenter_v3.pt`) | Candidate Exp 1 (`best_segmenter_v3_exp1_lr1e4.pt`) | Delta |
|---|---|---|---|
| **Best Epoch** | 2 | **29** | +27 epochs |
| **Dice** | 0.8279 | **0.8750** | **+0.0471** |
| **IoU** | 0.7063 | **0.7778** | **+0.0715** |
| **Precision** | 0.7987 | **0.8891** | **+0.0904** |
| **Recall** | 0.8593 | **0.8613** | **+0.0020** |
| **Pixel Accuracy** | 0.9528 | **0.9675** | **+0.0147** |
| **TP Pixels** | 1,844,038 | 1,848,513 | +4,475 |
| **FP Pixels** | 464,658 | **230,489** | **-234,169 (-50.4%)** |
| **FN Pixels** | 302,044 | 297,569 | -4,475 |
| **TN Pixels** | 13,642,188 | 13,876,357 | +234,169 |

---

## 5. Untouched Independent Test Set Evaluation

### 5.1 Segmenter Full-Scene Independent Test (20 Scenes, 83,886,080 Pixels)
Because segmenter training used only 120 of the 1,200 available scenes, an untouched test partition of 20 complete full scenes (`val_scenes[21:41]`, $2048 \times 2048$ dual-pol images) was evaluated using tiled inference (stride=384, patch=512, threshold=0.50):

| Metric | Baseline (Epoch 2) | Candidate Exp 1 (Epoch 29) | Operational Impact |
|---|---|---|---|
| **Full-Scene Macro Dice** | 0.5945 | **0.8390** | **+0.2445 Dice points** |
| **Full-Scene IoU** | 0.4230 | **0.7227** | **+0.2997 IoU points** |
| **Full-Scene Precision** | 0.4509 | **0.8013** | **+0.3504 Precision points** |
| **Full-Scene Recall** | 0.8722 | **0.8804** | **+0.0082 Recall points** |
| **False Positive Pixels** | 1,449,897 | **297,985** | **-1,151,912 FP pixels (-79.4%)** |
| **Inference Runtime / Scene** | 925.6 ms | **808.0 ms** | -117.6 ms (Faster) |

**Key Finding:** On untouched full scenes, the baseline suffered severe open-ocean speckle false positive accumulation, collapsing precision to 45.09% and full-scene Dice to 0.5945. The candidate segmenter suppresses 79.4% of open-ocean false alarms, maintaining 80.1% precision and 83.9% Dice on complete $2048 \times 2048$ SAR acquisitions.

---

### 5.2 Classifier Independent Test Set Status

> [!WARNING]
> **Independent Test Set Data Absence:**  
> An untouched independent test set for the classifier is **currently absent from local storage**.

#### Explanation:
- All 1,200 oil scenes in Part I are partitioned between `cls_train_v3` (983 scenes) and `cls_val_v3` (217 scenes).
- The official Zenodo Part III dataset (`https://zenodo.org/records/13761290`, `Test_Images.7z` and `Test_Mask.7z`), containing 450 held-out test scenes (150 Oil, 150 Clean, 150 Look-alike), is not present on disk at `D:/Test/Images`.
- The 40-scene benchmark gallery cannot serve as an independent test set because 8 of its scenes overlap with `cls_train_v3` (as documented in `BASELINE_BACKUP.md`).

#### Data Needed to Run the Classifier Independent Test:
To construct a completely uncompromised classifier test set, download Zenodo Part III:
```bash
python -m ml.download_zenodo --part 3 --dest D:/Test
```
Once downloaded, the candidate classifier will be evaluated on the 450 unseen Part III scenes without any fine-tuning.

---

## 6. Investigation: Why Classifier Metrics Improved Dramatically

The jump in classifier metrics from 31.5% recall / 47.1% F1 to 96.2% recall / 97.7% F1 is explainable by two mathematical and methodological factors, rather than data leakage:

1. **Resolution of Negative Gradient Asymmetry:**
   The baseline used $\alpha=0.25$ in Focal Loss on an already negative-skewed sampling distribution. The model received $3\times$ heavier penalty for false alarms than misses. It learned to output large negative logits, compressing oil patch probabilities below 0.16. Correcting $\alpha$ to $0.50$ restores symmetric gradient scale ($1:1$), allowing logits to calibrate to true class prevalence.
2. **Hard-Negative Ship-Wake Mineralization:**
   Ship wakes were previously only 10% of training samples ($242$ total patches). Increasing sampling weight to $30\%$ allowed the convolutional filters to learn the sharp contrast between Kelvin wake envelopes and diffuse viscoelastic oil slicks, dropping ship-wake FPR from $37.33\%$ to $2.67\%$.

---

## 7. Discrepancy Report & Promotion Recommendation

### Discrepancy Summary

| Issue | Severity | Audit Finding |
|---|---|---|
| **Duplicate Scene Files in Zenodo** | Medium | `00356.tif` $\equiv$ `00357.tif` and `00007.tif` $\equiv$ `01339.tif` are identical files in Zenodo Part I. They leaked 26 patches (0.54%) across splits. Deduplication proved this had zero impact on metrics ($<0.001$ F1 delta). |
| **Missing Zenodo Part III Test Set** | High | `D:/Test` is unpopulated. Final generalization claims for the classifier must await Part III download. |
| **Segmenter Baseline Under-training** | Critical | Baseline checkpoint stopped prematurely at epoch 2 due to aggressive LR ($5\times 10^{-4}$) and short patience (6). Fixed by LR=$1\times 10^{-4}$, patience=12. |

### Promotion Recommendation

1. **Segmenter (`best_segmenter_v3_exp1_lr1e4.pt`):**
   - **RECOMMENDATION: PROMOTE TO PRODUCTION CANDIDATE.**
   - Demonstrated +0.0471 Dice gain on validation crops and **+0.2445 Dice gain (0.595 $\rightarrow$ 0.839)** on 20 untouched independent full scenes. Reduced full-scene false alarms by 79.4%.
   - Original checkpoint remains archived.
2. **Classifier (`best_dualpol_classifier_exp2_sampling_sw30.pt`):**
   - **RECOMMENDATION: HOLD IN STAGING.**
   - While validation metrics are outstanding (Oil Recall: 96.24%, Ship-Wake FPR: 2.67%, F1: 0.9767), formal promotion to production should be deferred until evaluation against Zenodo Part III test scenes is complete.

---

*Report generated and validated by SpillSense Phase 3 Audit Suite.*  
*Artifact: `ml/benchmarks/failure_analysis/PHASE3_VALIDATION_AUDIT.md`*
