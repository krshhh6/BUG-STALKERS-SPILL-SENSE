# SpillSense SIH-26143 — Segmenter Production Release Report

**Release Date:** 2026-10-14  
**Release Author:** Antigravity Pair-Programming Assistant  
**Git Branch:** `improvement/phase0-backup` (HEAD: `d6e7730`)  
**Component:** SpillSegNet U-Net Segmenter (`oil_segmenter.onnx`)  
**Release Status:** **SUCCESSFULLY DEPLOYED TO PRODUCTION**

---

## 1. Executive Summary

The improved oil spill segmentation model candidate `best_segmenter_v3_exp1_lr1e4.pt` (epoch 29) has successfully passed all verification gates, interface validations, numerical equivalence checks, and full-scene smoke tests. It has been promoted to production at:
`frontend/public/models/oil_segmenter.onnx`.

The previous production baseline has been preserved in multiple backup locations, and single-command rollback instructions are tested and documented below.

---

## 2. Checkpoint & Model Provenance

### Candidate Checkpoint Verification

| Property | Value |
|---|---|
| **Checkpoint Path** | [`ml/checkpoints/best_segmenter_v3_exp1_lr1e4.pt`](file:///d:/Spill%20Sense/SIH-26143-OIL-Spill/ml/checkpoints/best_segmenter_v3_exp1_lr1e4.pt) |
| **Model Architecture** | `SpillSegNet` (Compact Dual-Polarization SAR U-Net) |
| **Training Hyperparameters** | Learning Rate: $1\times 10^{-4}$ (AdamW), Patience: 12, Epochs: 29, Loss: $0.6 \cdot \text{Dice} + 0.4 \cdot \text{BCE}$ |
| **Expected SHA-256 Hash** | `38DD8341C2CD0D1DD0DDB5082130C96B675CD991F8FB692E7B206D1831B3584D` |
| **Actual SHA-256 Hash** | `38DD8341C2CD0D1DD0DDB5082130C96B675CD991F8FB692E7B206D1831B3584D` |
| **Status** | **VERIFIED BIT-FOR-BIT** |

---

## 3. Production Model Backup Registry

Prior to overwriting the active model, the original production ONNX was copied and verified across two locations:

| Backup Role | File Path | File Size | SHA-256 Cryptographic Hash |
|---|---|---|---|
| **Frontend Public Backup** | [`frontend/public/models/oil_segmenter_baseline_backup.onnx`](file:///d:/Spill%20Sense/SIH-26143-OIL-Spill/frontend/public/models/oil_segmenter_baseline_backup.onnx) | 6,367,937 bytes | `8E58A9AF1FB3D3F0C94A79DAAA17DA2015910F82D2A74EBCFE02B0CC60BC3DE9` |
| **ML Checkpoints Archive** | [`ml/checkpoints/backup_oil_segmenter_baseline.onnx`](file:///d:/Spill%20Sense/SIH-26143-OIL-Spill/ml/checkpoints/backup_oil_segmenter_baseline.onnx) | 6,367,937 bytes | `8E58A9AF1FB3D3F0C94A79DAAA17DA2015910F82D2A74EBCFE02B0CC60BC3DE9` |
| **Original PyTorch Source** | [`ml/checkpoints/best_segmenter_v3.pt`](file:///d:/Spill%20Sense/SIH-26143-OIL-Spill/ml/checkpoints/best_segmenter_v3.pt) | 6,432,607 bytes | `9C5901016C19CB93B8B492512936D8726E1B988F32AF8E392A5225C260C08B08` |

Both backup copies match the pre-promotion production model hash exactly.

---

## 4. Staging Export & Interface Validation

The candidate checkpoint was exported to a staging location (`ml/checkpoints/temp_oil_segmenter_candidate.onnx`) with:
- **ONNX Opset:** 14
- **Constant Folding:** Enabled
- **Input Tensor Contract:** `input` $\rightarrow$ `['batch', 2, 512, 512]` (`float32`)
- **Output Tensor Contract:** `output` $\rightarrow$ `['batch', 1, 512, 512]` (`float32`)
- **Dynamic Axes:** Batch dimension (`{0: 'batch'}`) on both input and output

### Numerical Parity Test (PyTorch vs. ONNX Runtime)
- Evaluated on random Gaussian input tensors $\sim \mathcal{N}(0, 1)$.
- PyTorch probabilities: $\sigma(\text{model}(x))$
- ONNX probabilities: $\sigma(\text{ort\_session.run}(x))$
- **Maximum Absolute Discrepancy (MAE):** **$3.16 \times 10^{-6}$** (well below the $10^{-5}$ threshold).
- **Status:** **PASSED**

---

## 5. Representative-Scene Full-Scale Smoke Tests

Full $2048 \times 2048$ dual-polarization SAR scenes were preprocessed via [`normalize_sar_dualpol`](file:///d:/Spill%20Sense/SIH-26143-OIL-Spill/ml/preprocess.py) and evaluated with the production sliding window tiled inference (patch size 512, stride 384):

### Scene 1: `00492.tif` (87,206 GT Oil Pixels, 2.08% Coverage)
- **PyTorch Reference Full-Scene Dice:** **0.8980** | **IoU:** **0.8149**
- **ONNX Runtime Full-Scene Dice:** **0.8980** | **IoU:** **0.8149**
- **Full-Scene Maximum Probability Discrepancy:** **$9.80 \times 10^{-6}$**
- **Status:** **PASSED**

### Scene 2: `00563.tif` (202,076 GT Oil Pixels, 4.82% Coverage)
- **PyTorch Reference Full-Scene Dice:** **0.8746** | **IoU:** **0.7771**
- **ONNX Runtime Full-Scene Dice:** **0.8746** | **IoU:** **0.7771**
- **Full-Scene Maximum Probability Discrepancy:** **$9.95 \times 10^{-6}$**
- **Status:** **PASSED**

---

## 6. Deployed Production Model Specifications

| Attribute | Specification |
|---|---|
| **Production File Path** | [`frontend/public/models/oil_segmenter.onnx`](file:///d:/Spill%20Sense/SIH-26143-OIL-Spill/frontend/public/models/oil_segmenter.onnx) |
| **Deployed SHA-256 Hash** | `D464E6CEF8DC47B7AFDD0E7F0AE1068AE92798DF3154004BB771B7ED317D4A0E` |
| **Deployed File Size** | 6,367,937 bytes (6.07 MB) |
| **Inference Frameworks** | ONNX Runtime Web / WebAssembly (`ort-wasm.wasm`) & PyTorch CPU/CUDA |
| **Full-Scene Micro Dice** | **0.8390** (Baseline was 0.5945; **+0.2445 gain**) |
| **Open-Ocean False Positives** | **297,985 pixels** (Baseline was 1,449,897; **-79.4% false alarms**) |
| **Inference Latency** | **584.1 ms** per $2048 \times 2048$ scene on RTX 4060 GPU (Identical to baseline) |

---

## 7. Tested Rollback Instructions

If any operational issue is detected, execute either of the following tested rollback procedures:

### Option 1: Instant Single-Command File Rollback
```powershell
Copy-Item "frontend/public/models/oil_segmenter_baseline_backup.onnx" -Destination "frontend/public/models/oil_segmenter.onnx" -Force
```

### Option 2: Python Programmatic Rollback & Hash Check
```python
import shutil, hashlib
from pathlib import Path

bak = Path("frontend/public/models/oil_segmenter_baseline_backup.onnx")
prod = Path("frontend/public/models/oil_segmenter.onnx")
shutil.copy2(bak, prod)
h = hashlib.sha256(prod.read_bytes()).hexdigest().upper()
print(f"Rollback complete. Restored model hash: {h}")
assert h == "8E58A9AF1FB3D3F0C94A79DAAA17DA2015910F82D2A74EBCFE02B0CC60BC3DE9"
```

---

*Report generated and validated by `ml/scripts/verify_and_promote_segmenter.py`.*  
*Artifact: `SEGMENTER_RELEASE_REPORT.md`*
