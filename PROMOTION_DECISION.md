# SpillSense SIH-26143 — Model Promotion Decision & Governance

**Date:** 2026-10-14  
**Audit Author:** Antigravity Pair-Programming Assistant  
**Git Branch:** `improvement/phase0-backup` (HEAD: `d6e7730`)  
**Scope:** Production Promotion Review for SpillSense ML Checkpoints  
**Preceding Artifacts:** `PHASE3_VALIDATION_AUDIT.md`, `PHASE4_FINAL_VALIDATION.md`, `segmenter_per_scene_results.csv`, `classifier_independent_test_results.json`

---

## 1. Summary of Promotion Decisions

| Model Component | Candidate Checkpoint | Status | Promotion Decision | Primary Rationale |
|---|---|---|---|---|
| **Oil Spill Segmenter** | `best_segmenter_v3_exp1_lr1e4.pt` (Epoch 29) | Leading Candidate | **APPROVED FOR PROMOTION** | Provenance verified on 20 independent full scenes (83.88M pixels). Micro Dice jumped from 0.5945 to 0.8390 (+0.2445); false alarms reduced by 79.4%; tile seam and faint slick tests passed. |
| **Dual-Pol Classifier** | `best_dualpol_classifier_exp2_sampling_sw30.pt` (Epoch 4) | Staging Candidate | **HOLD IN STAGING** | Zenodo Part III test data (450 scenes) is absent from local disk. Per Rule B.6, promotion is deferred until Part III download and independent evaluation complete. |

---

## 2. Model Integrity & Checkpoint Registry

Every checkpoint is cryptographically anchored by SHA-256 hash. No weights have been retrained or modified on test data.

```json
{
  "checkpoints": {
    "baseline_segmenter": {
      "path": "ml/checkpoints/best_segmenter_v3.pt",
      "sha256": "9C5901016C19CB93B8B492512936D8726E1B988F32AF8E392A5225C260C08B08",
      "architecture": "SpillSegNet (U-Net)",
      "epoch": 2,
      "status": "Archived Production Baseline"
    },
    "candidate_segmenter": {
      "path": "ml/checkpoints/best_segmenter_v3_exp1_lr1e4.pt",
      "sha256": "38DD8341C2CD0D1DD0DDB5082130C96B675CD991F8FB692E7B206D1831B3584D",
      "architecture": "SpillSegNet (U-Net)",
      "epoch": 29,
      "status": "Promoted Production Candidate"
    },
    "baseline_classifier": {
      "path": "ml/checkpoints/best_dualpol_classifier.pt",
      "sha256": "262EF1A8D1F3FBCFBBEDBCC67AEE0C3C1FAF5995CC855CF7131D1FD14C78D902",
      "architecture": "DualPolOilSpillNet",
      "epoch": 7,
      "operating_threshold": 0.49,
      "status": "Active Production Classifier"
    },
    "candidate_classifier": {
      "path": "ml/checkpoints/best_dualpol_classifier_exp2_sampling_sw30.pt",
      "sha256": "014D209A19DBD5C42DA275648F8E5BCB748ABCFCB1562B91F0541FBDCCE0653E",
      "architecture": "DualPolOilSpillNet",
      "epoch": 4,
      "operating_threshold": 0.43,
      "status": "Frozen Staging Candidate"
    }
  }
}
```

---

## 3. Segmenter Promotion Review

### 3.1 Verification Checklist & Audit Findings

| Verification Gate | Requirement | Audit Result | Status |
|---|---|---|---|
| **Data Provenance** | 20 independent full-scene GeoTIFFs ($2048 \times 2048$) strictly held out from training and validation | Verified: `val_scenes[21:41]` from Zenodo Part I. Zero overlap with `train_scenes[:98]` or `val_scenes[:21]`. Zero overlap with Zenodo duplicate pairs. | **PASSED** |
| **Reproducibility** | Exact reproduction using tile size 512, stride 384, threshold 0.50 | Verified: Re-evaluated end-to-end via `ml/benchmarks/phase4_segmenter_validation.py`. Results match bit-for-bit with Phase 3 records. | **PASSED** |
| **Micro (Aggregate) Dice** | Substantial gain on pooled 83,886,080 pixels | Baseline: **0.5945** $\rightarrow$ Candidate: **0.8390** (+0.2445 gain). IoU improved from 0.4230 to **0.7227** (+0.2997). | **PASSED** |
| **False Alarm Suppression** | Reduction of open-ocean speckle false positives | Baseline: 1,449,897 FP pixels $\rightarrow$ Candidate: **297,985 FP pixels** (**-79.4% reduction**; 1,151,912 fewer false alarms). | **PASSED** |
| **Macro Metrics** | Improvement across unweighted per-scene distribution | Mean Dice: 0.7556 $\rightarrow$ **0.8219** (+0.0664). Median Dice: 0.8246 $\rightarrow$ **0.8619** (+0.0373). Std Dev: 0.1981 $\rightarrow$ **0.1598** (more consistent). | **PASSED** |
| **Small Spill Detection** | Performance on challenging small slicks ($<50\text{k}$ px / $<1\%$ scene area) | Mean Dice improved from 0.6820 to **0.7205** (+0.0385); Mean Precision jumped from 0.6610 to **0.7350** (+0.0740). Scene `oil_00682` Dice rose from 0.0284 to 0.2105 (7.4x gain). | **PASSED** |
| **Faint Slick Sensitivity** | Retention of low-contrast slicks ($<3.0$ dB anomaly) | Scene `oil_01013` (contrast 2.78 dB): Dice improved from 0.8214 to **0.8511**; Recall improved from 0.7127 to **0.7532**. | **PASSED** |
| **Tile Seam Uniformity** | Absence of border artifacts along 384-pixel stride seams | Seam error ratio is **0.922x** ($\le 1.0$). 128-pixel sliding window overlap produces smooth spatial ensembling with no seam artifacts. | **PASSED** |
| **Inference Runtime** | No latency regression on target hardware | Baseline: 577.9 ms $\rightarrow$ Candidate: **584.1 ms** (within 1% parity on RTX 4060 GPU). | **PASSED** |

### 3.2 Segmenter Promotion Verdict
**PROMOTION GRANTED.** Candidate `best_segmenter_v3_exp1_lr1e4.pt` demonstrates clear, unambiguous superiority over the baseline across every operational and statistical dimension, solving the critical open-ocean false-alarm failure mode of the baseline model while maintaining execution throughput.

---

## 4. Classifier Staging Review

### 4.1 Verification Checklist & Audit Findings

| Verification Gate | Requirement | Audit Result | Status |
|---|---|---|---|
| **Independent Test Data** | Official Zenodo Part III test set (450 scenes) on local disk | Absent from `D:/Test`. The archive is not present on disk. | **FAILED (ABSENT)** |
| **Downloader Integrity** | Documented `ml/download_zenodo.py` downloads Part III | Downloader contained obsolete URLs (`Images.7z` / `Mask.7z`) returning HTTP 404. Corrected to target `02_Test_images_and_ground_truth.7z` (9.18 GB). | **CORRECTED** |
| **Network Throughput** | Viable transfer speed from Zenodo repository | Network probe measured 0.051 MB/s (~50 hours transfer time). Transfer cannot complete in interactive session. | **BLOCKED** |
| **Compliance Rule B.6** | Stop and report limitation; do not substitute validation scores | Halted evaluation immediately. Stated limitation transparently in `classifier_independent_test_results.json`. Validation scores were NOT substituted. | **PASSED** |
| **Candidate Quality (Val Split)** | High recall on validation set | Candidate Exp 1B achieved 96.24% Oil Recall, 97.67% F1, and reduced Ship-Wake FPR from 37.33% to 2.67%. | **VERIFIED (VAL)** |
| **Threshold Freeze** | Operating threshold frozen prior to test evaluation | Frozen at **0.43** (Exp 1B) and **0.49** (Baseline). Zero tuning on test set. | **PASSED** |

### 4.2 Classifier Staging Verdict
**HOLD IN STAGING.** While candidate `best_dualpol_classifier_exp2_sampling_sw30.pt` demonstrates remarkable performance on the 4,818-patch validation partition, promotion to production is deferred until Zenodo Part III is downloaded and evaluated without tuning. Active production remains `best_dualpol_classifier.pt`.

---

## 5. Production Promotion & Rollback Instructions

### 5.1 Segmenter Promotion Procedure

To promote the candidate segmenter `best_segmenter_v3_exp1_lr1e4.pt` to the primary ONNX artifact for browser and backend deployment:

```bash
# Step 1: Backup current production ONNX
cp frontend/public/models/oil_segmenter.onnx frontend/public/models/oil_segmenter_baseline_backup.onnx

# Step 2: Export candidate checkpoint to production ONNX
python -c "
import torch
from pathlib import Path
from ml.models import get_segmenter
from ml.config import CONFIG

device = torch.device('cpu')
ckpt = torch.load('ml/checkpoints/best_segmenter_v3_exp1_lr1e4.pt', map_location=device, weights_only=False)
model = get_segmenter(2).to(device)
model.load_state_dict(ckpt['model_state'])
model.eval()

dummy = torch.randn(1, 2, 512, 512)
out_path = Path('frontend/public/models/oil_segmenter.onnx')
torch.onnx.export(
    model, dummy, str(out_path),
    export_params=True, opset_version=14, do_constant_folding=True,
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
)
print(f'Successfully promoted candidate to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)')
"

# Step 3: Verify ONNX Runtime loads and executes without error
python -c "
import onnxruntime as ort, numpy as np
sess = ort.InferenceSession('frontend/public/models/oil_segmenter.onnx', providers=['CPUExecutionProvider'])
dummy = np.zeros((1, 2, 512, 512), dtype=np.float32)
out = sess.run(None, {'input': dummy})
print('ONNX Verification Passed. Output shape:', out[0].shape)
"
```

### 5.2 Segmenter Rollback Procedure

If any regression is detected in production, execute the single-command rollback:

```bash
# Rollback segmenter ONNX to baseline backup
cp frontend/public/models/oil_segmenter_baseline_backup.onnx frontend/public/models/oil_segmenter.onnx
```

Or re-export directly from the original baseline `.pt` checkpoint:

```bash
python -c "
import torch
from pathlib import Path
from ml.models import get_segmenter

device = torch.device('cpu')
ckpt = torch.load('ml/checkpoints/best_segmenter_v3.pt', map_location=device, weights_only=False)
model = get_segmenter(2).to(device)
model.load_state_dict(ckpt['model_state'])
model.eval()

dummy = torch.randn(1, 2, 512, 512)
out_path = Path('frontend/public/models/oil_segmenter.onnx')
torch.onnx.export(
    model, dummy, str(out_path),
    export_params=True, opset_version=14, do_constant_folding=True,
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
)
print('Rollback complete. Restored baseline checkpoint best_segmenter_v3.pt to oil_segmenter.onnx')
"
```

---

### 5.3 Classifier Staging Switch & Future Promotion Procedure

When Zenodo Part III download completes:

```bash
# Step 1: Download Part III archive (run via background task or dedicated connection)
python -m ml.download_zenodo --part 3 --dest D:/Test

# Step 2: Run independent evaluation on the 450 untouched Part III scenes
python -m ml.evaluation.evaluate_part3_independent \
  --baseline ml/checkpoints/best_dualpol_classifier.pt \
  --candidate ml/checkpoints/best_dualpol_classifier_exp2_sampling_sw30.pt

# Step 3: If candidate meets criteria (F1 >= 0.95, Ship-Wake FPR <= 5%), export candidate ONNX:
python -c "
import torch
from pathlib import Path
from ml.models import get_classifier

device = torch.device('cpu')
ckpt = torch.load('ml/checkpoints/best_dualpol_classifier_exp2_sampling_sw30.pt', map_location=device, weights_only=False)
model = get_classifier(2).to(device)
model.load_state_dict(ckpt['model_state'])
model.eval()

dummy = torch.randn(1, 2, 400, 400)
out_path = Path('frontend/public/models/oil_classifier.onnx')
torch.onnx.export(
    model, dummy, str(out_path),
    export_params=True, opset_version=14, do_constant_folding=True,
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
)
print('Classifier promoted to production.')
"
```

---

*SpillSense Phase 4 Model Promotion Governance Record.*  
*Artifact: `PROMOTION_DECISION.md`*
