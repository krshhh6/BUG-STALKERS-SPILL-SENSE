# SpillSense SIH-26143 — Classifier Independent Validation & Staging Audit Report

**Date:** 2026-10-14  
**Audit Author:** Antigravity Pair-Programming Assistant  
**Git Repository:** [`https://github.com/krshhh6/BUG-STALKERS-SPILL-SENSE.git`](https://github.com/krshhh6/BUG-STALKERS-SPILL-SENSE)  
**Local Workspace:** `D:\Spill Sense\SIH-26143-OIL-Spill`  
**Git Commit / Head:** `f6a35c0` on branches `main` and `improvement/phase0-backup`  
**Audited Component:** DualPolOilSpillNet Classifier Checkpoints & Test Pipeline  
**Promotion Verdict:** **NOT ELIGIBLE FOR PROMOTION — REMAINS FROZEN IN STAGING**

---

## Executive Summary & Governance Stance

In accordance with the strict verification protocols of SpillSense Phase 4:
1. **Segmenter (Phase 4 Completed):** The improved segmenter (`best_segmenter_v3_exp1_lr1e4.pt`, epoch 29) was verified on 20 independent full scenes (83.88M pixels, 0.8390 Micro Dice, 79.4% false-alarm reduction) and deployed as `frontend/public/models/oil_segmenter.onnx`. It remains untouched and in production.
2. **Classifier (Candidate Held in Staging):** The classifier candidate `best_dualpol_classifier_exp2_sampling_sw30.pt` cannot be promoted because the independent test set (Zenodo Part III) download is currently incomplete (12.7% transferred).
3. **Ship-Wake Metric Limitation Identified:** The official Zenodo Part III dataset comprises only 3 classes (Oil, No-Oil, Look-alike) and lacks explicit ground-truth vessel wake labels. Consequently, ship-wake FPR is **not measurable on this test set**, requiring a separate labelled wake challenge dataset.
4. **Promotion Decision:** The classifier candidate **strictly remains in staging**. Active production remains `best_dualpol_classifier.pt` / `frontend/public/models/oil_classifier.onnx`.

---

## 1. Test Dataset Readiness & Download Audit

### 1.1 Live Download Process State
- **Process Worker:** Dedicated background worker [`ml/scripts/download_part3_resumable.py`](file:///d:/Spill%20Sense/SIH-26143-OIL-Spill/ml/scripts/download_part3_resumable.py) (Daemon task `task-561`).
- **Worker Process State:** **ACTIVE & STREAMING** (Single worker process, zero competing or duplicate downloaders).
- **Target Remote Archive:** Zenodo Record `13761290`, `02_Test_images_and_ground_truth.7z` (9,859,650,011 bytes / 9.86 GB).
- **Target Local Path:** `D:\Test\02_Test_images_and_ground_truth.7z.part`
- **Expected Archive MD5:** `5dce64cd7ff9d80189d13504bd3bcbf5`

### 1.2 Live Progress Snapshot
- **Bytes Downloaded:** **1,251,999,744 bytes (1.25 GB / 12.70%)**
- **Bytes Remaining:** **8,607,650,267 bytes (8.61 GB)**
- **Transfer Throughput:** ~0.08 – 0.56 MB/s (variable depending on Zenodo CDN throttling)
- **Current ETA:** ~23.5 – 30.0 hours
- **Resumption & Retry Audit:** At 22:15:54 local time, a remote connection dropped (`IncompleteRead`). The worker automatically invoked exponential backoff, sent HTTP Range header `bytes=1232076800-`, received HTTP 206 Partial Content, and resumed streaming without losing a single downloaded byte.

### 1.3 Data Integrity Gate
- The final archive `02_Test_images_and_ground_truth.7z` does not yet exist on disk (only the `.part` staging file exists).
- MD5 checksum verification **cannot be performed until 100% of bytes are transferred**.
- **Rule Enforcement:** Under no circumstances will an incomplete or unverified partial archive be extracted or treated as valid test data.

---

## 2. Frozen Independent Evaluation Harness Audit

The independent evaluation script [`ml/scripts/evaluate_part3_independent.py`](file:///d:/Spill%20Sense/SIH-26143-OIL-Spill/ml/scripts/evaluate_part3_independent.py) was reviewed and tested live:

```
[TEST DATASET] Found 0 TIFF files in D:\Test
  Discovered Oil Scenes:       0
  Discovered Clean Scenes:     0
  Discovered Look-alike Scenes: 0

[BLOCKED] Only 0 scenes found in D:/Test. Archive download or extraction is still in progress.
  Current download status can be monitored via ml/logs/part3_download_status.json
```

### Safety Features Verified:
1. **Execution Blocked on Missing Data:** The harness detects that `D:/Test` contains 0 extracted scenes and safely halts with return code 0, rather than crashing or substituting validation scores.
2. **Frozen Operating Thresholds:**
   - Baseline: Evaluated strictly at threshold **0.49**.
   - Candidate: Evaluated strictly at threshold **0.43**.
   - **Zero tuning, zero threshold sweeps, and zero weight updates are permitted on test data.**
3. **Ground-Truth Mapping:** Strict 1-to-1 pairing of TIFF acquisitions with authentic Zenodo binary masks.

---

## 3. Metric Triangulation: Separation of Evidence

To adhere to scientific integrity standards, results are partitioned into three distinct classes:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        EVIDENCE PARTITIONING                           │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Independently Reproduced Results (Verified in Phase 4)             │
│ 2. Results Merely Reported by Earlier Runs (Validation Split)          │
│ 3. Metrics that Remain Unverified or Unmeasurable                      │
└────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Class 1: Independently Reproduced Results (Verified in Phase 4)
*Evaluated on 20 untouched, independent full scenes (`val_scenes[21:41]`, 83,886,080 pixels total) at patch size 512, stride 384, threshold 0.50:*

| Model | Evaluation Target | Micro (Aggregate) Dice | Micro IoU | Precision | Recall | False Positive Pixels |
|---|---|---|---|---|---|---|
| **Baseline Segmenter** (Epoch 2) | 20 Untouched Full Scenes | 0.5945 | 0.4230 | 0.4509 | 0.8722 | 1,449,897 px |
| **Candidate Segmenter** (Epoch 29) | 20 Untouched Full Scenes | **0.8390** (+0.2445) | **0.7227** (+0.2997) | **0.8013** (+0.3504) | **0.8804** | **297,985 px (-79.4%)** |

*Status: **PASSED & PROMOTED TO PRODUCTION** ([`SEGMENTER_RELEASE_REPORT.md`](file:///d:/Spill%20Sense/SIH-26143-OIL-Spill/SEGMENTER_RELEASE_REPORT.md)).*

---

### 3.2 Class 2: Results Reported on Curated Validation Split
*Evaluated on `cls_val_v3_zenodo_gt_dualpol.npz` (4,818 patches from 217 scenes; 3,272 Oil, 1,082 Clean, 389 Look-alike, 75 heuristic Ship Wakes):*

| Metric | Baseline (`best_dualpol_classifier.pt`) | Candidate Exp 1B (`best_dualpol_classifier_exp2...`) | Delta |
|---|---|---|---|
| **Operating Threshold** | 0.49 (frozen) | 0.43 (frozen) | -0.06 |
| **Accuracy** | 52.05% (2,508 / 4,818) | **96.89% (4,668 / 4,818)** | +44.84% |
| **Precision** | 0.9389 (1,029 / 1,096) | **0.9915 (3,149 / 3,176)** | +0.0526 |
| **Recall (Oil)** | 31.45% (1,029 / 3,272) | **96.24% (3,149 / 3,272)** | **+64.79%** |
| **F1 Score** | 0.4712 | **0.9767** | **+0.5055** |
| **ROC-AUC** | 0.6985 | **0.9959** | +0.2974 |
| **PR-AUC** | 0.5778 | **0.9736** | +0.3958 |
| **Ship-Wake FPR** (Heuristic Split) | 37.33% (28 / 75 FPs) | **2.67% (2 / 75 FPs)** | **-34.66%** |
| **Look-Alike FPR** | 6.94% (27 / 389 FPs) | **1.80% (7 / 389 FPs)** | -5.14% |
| **Clean Ocean FPR** | 1.11% (12 / 1,082 FPs) | 1.66% (18 / 1,082 FPs) | +0.55% |
| **Confusion Matrix** | `TP: 1029, FP: 67, FN: 2243, TN: 1479` | `TP: 3149, FP: 27, FN: 123, TN: 1519` | -2,120 FNs |

*Note: While these validation results confirm that resolving the negative focal loss gradient asymmetry ($\alpha=0.50$) and increasing ship-wake sampling ($w=0.30$) fixed the baseline's 31.5% recall failure, **they do not substitute for independent test set results**.*

---

### 3.3 Class 3: Metrics that Remain Unverified or Unmeasurable

| Metric | Target Dataset | Status | Reason & Analysis |
|---|---|---|---|
| **Independent Oil Recall** | Zenodo Part III (150 Oil Scenes) | **UNVERIFIED** | Dataset download is at 12.7% (1.25 GB / 9.86 GB). Cannot evaluate until transfer and extraction complete. |
| **Independent F1 Score** | Zenodo Part III (450 Scenes) | **UNVERIFIED** | Awaiting complete download and MD5 verification. |
| **Independent Clean FPR** | Zenodo Part III (150 Clean Scenes) | **UNVERIFIED** | Awaiting complete download. |
| **Independent Look-Alike FPR** | Zenodo Part III (150 Look-Alike Scenes) | **UNVERIFIED** | Awaiting complete download. |
| **Ship-Wake FPR on Part III** | Zenodo Part III | **NOT MEASURABLE** | **Critical Ground-Truth Limitation:** Zenodo Part III provides only 3 scene classes: Oil, No-Oil, and Look-alike. It contains **no vessel AIS tracking data, vessel coordinates, or annotated Kelvin wake labels**. Wake FPR cannot be measured on Part III without heuristic synthesis. |

---

## 4. Application of Predefined Promotion Gates

The candidate classifier `best_dualpol_classifier_exp2_sampling_sw30.pt` was evaluated against the mandatory promotion gates:

```
Gate 1: Independent Oil Recall ≥ 90.0%
Gate 2: Ship-Wake FPR ≤ 5.0% (measured on a valid labelled wake set)
Gate 3: Independent Overall F1 ≥ 0.950
```

### Gate Compliance Evaluation:

| Promotion Gate | Requirement | Current Measured Value | Gate Status |
|---|---|---|---|
| **Gate 1: Oil Recall** | $\ge 90.0\%$ on independent test set | **Unverified** (Validation: 96.24%) | **NOT MET (PENDING DATA)** |
| **Gate 2: Ship-Wake FPR** | $\le 5.0\%$ on labelled wake set | **Not Measurable on Part III** (Validation: 2.67%) | **NOT MET (UNMEASURABLE ON PART III)** |
| **Gate 3: Overall F1** | $\ge 0.950$ on independent test set | **Unverified** (Validation: 0.9767) | **NOT MET (PENDING DATA)** |

### Governance Verdict:
> [!IMPORTANT]
> **GATE FAILURE / DEFERRAL:**  
> The candidate classifier **FAILS TO PASS THE PROMOTION GATES** at this time because:
> 1. The independent test data transfer is incomplete (12.7%).
> 2. Gate 2 (Ship-Wake FPR) is mathematically unmeasurable on Zenodo Part III due to the absence of authentic vessel wake labels in the published Zenodo archive.

**DECISION: RETAIN CANDIDATE IN STAGING.**  
Under zero circumstances may a model be promoted based on validation scores or unverified test metrics.

---

## 5. Recommended Ship-Wake Challenge Set Protocol

To resolve the unmeasurability of Gate 2 on Zenodo Part III:
1. **Challenge Dataset Construction:** Construct an isolated benchmark challenge set using raw Copernicus Sentinel-1 IW GRD acquisitions correlated with live AIS transponder feeds (MMSI, speed $> 8$ knots, heading) across the Mumbai High / Gulf of Khambhat shipping lanes.
2. **True Kelvin Wake Annotation:** Extract 200 real ship-wake patches containing verified vessel returns and associated divergent/transverse wake envelopes in oil-free waters.
3. **Wake Challenge Gate:** Evaluate the candidate classifier on this dedicated wake dataset to confirm ship-wake false positive rate $\le 5.0\%$.

---

## 6. Model & Artifact Preservation Registry

All production and candidate models remain untouched, safely archived, and verified:

```json
{
  "active_production_models": {
    "segmenter_onnx": {
      "path": "frontend/public/models/oil_segmenter.onnx",
      "sha256": "D464E6CEF8DC47B7AFDD0E7F0AE1068AE92798DF3154004BB771B7ED317D4A0E",
      "status": "PROMOTED & ACTIVE"
    },
    "classifier_onnx": {
      "path": "frontend/public/models/oil_classifier.onnx",
      "sha256": "B002352F633F55928CD03603B6434EF41DEE88B9E3A9C8D50FDEFE627DEEB16C",
      "status": "BASELINE ACTIVE (UNTOUCHED)"
    }
  },
  "staged_candidates": {
    "classifier_exp2": {
      "path": "ml/checkpoints/best_dualpol_classifier_exp2_sampling_sw30.pt",
      "sha256": "014D209A19DBD5C42DA275648F8E5BCB748ABCFCB1562B91F0541FBDCCE0653E",
      "frozen_threshold": 0.43,
      "status": "HELD IN STAGING"
    }
  },
  "archived_backups": {
    "segmenter_baseline_backup": {
      "path": "frontend/public/models/oil_segmenter_baseline_backup.onnx",
      "sha256": "8E58A9AF1FB3D3F0C94A79DAAA17DA2015910F82D2A74EBCFE02B0CC60BC3DE9",
      "status": "VERIFIED BASELINE BACKUP"
    },
    "classifier_baseline_pt": {
      "path": "ml/checkpoints/best_dualpol_classifier.pt",
      "sha256": "262EF1A8D1F3FBCFBBEDBCC67AEE0C3C1FAF5995CC855CF7131D1FD14C78D902",
      "frozen_threshold": 0.49,
      "status": "VERIFIED BASELINE CHECKPOINT"
    }
  }
}
```

---

*Report validated by SpillSense Phase 4 Final Audit Suite.*  
*Artifact: `CLASSIFIER_INDEPENDENT_VALIDATION_REPORT.md`*
