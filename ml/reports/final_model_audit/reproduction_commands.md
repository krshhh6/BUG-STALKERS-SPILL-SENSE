# SpillSense (SIH-26143) — Reproduction Commands & Execution Manual

**Target:** Deterministic reproduction of verified experiments from a clean checkout  
**Environment:** Python 3.11.9, PyTorch 2.6.0+cu124, ONNX Runtime 1.29.0, CUDA 12.4  
**Workspace Root:** `D:\Spill Sense\SIH-26143-OIL-Spill`

---

## 1. Environment Setup

```bash
# Clone and checkout verified branch
git clone https://github.com/krshhh6/BUG-STALKERS-SPILL-SENSE.git
cd BUG-STALKERS-SPILL-SENSE
git checkout main

# Create Python 3.11 virtual environment
python -m venv venv
venv\Scripts\activate

# Install exact dependencies
pip install -r backend/requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install onnxruntime-gpu
```

---

## 2. Model Integrity & Cryptographic Hash Verification

Verify all active models, checkpoints, and baseline backups match immutable SHA-256 signatures:

```powershell
Get-FileHash frontend/public/models/oil_segmenter.onnx, `
             frontend/public/models/oil_segmenter_baseline_backup.onnx, `
             frontend/public/models/oil_classifier.onnx, `
             ml/checkpoints/best_dualpol_classifier.pt, `
             ml/checkpoints/best_dualpol_classifier_exp2_sampling_sw30.pt, `
             ml/checkpoints/best_segmenter_v3_exp1_lr1e4.pt | Format-Table Path, Hash
```

Expected Hashes:
- Production Segmenter ONNX: `D464E6CEF8DC47B7AFDD0E7F0AE1068AE92798DF3154004BB771B7ED317D4A0E`
- Baseline Segmenter Backup: `8E58A9AF1FB3D3F0C94A79DAAA17DA2015910F82D2A74EBCFE02B0CC60BC3DE9`
- Production Classifier ONNX: `B002352F633F55922F5B4ED44480EB9E560425B4E55810947A7F2734C435C530`
- Baseline Classifier PT: `262EF1A8D1F3FBCFBBEDBCC67AEE0C3C1FAF5995CC855CF7131D1FD14C78D902`
- Candidate Classifier Exp 1B PT: `014D209A19DBD5C42DA275648F8E5BCB748ABCFCB1562B91F0541FBDCCE0653E`
- Candidate Segmenter PT: `38DD8341C2CD0D1DD0DDB5082130C96B675CD991F8FB692E7B206D1831B3584D`

---

## 3. Full-Scene Segmenter Independent Reproduction (20 Held-Out Scenes)

Rerun the evaluation of baseline (`best_segmenter_v3.pt`, epoch 2) vs candidate (`best_segmenter_v3_exp1_lr1e4.pt`, epoch 29) on the 20 untouched scenes (`val_scenes[21:41]`):

```bash
python ml/benchmarks/phase4_segmenter_validation.py
```

Outputs:
- `segmenter_validation_summary.json`
- `segmenter_per_scene_results.csv` (identical to `ml/reports/final_model_audit/per_scene_segmentation_metrics.csv`)
- Verified Headline Metrics:
  - Micro Dice: Baseline `0.5945`, Candidate `0.8390` (+0.2445)
  - Macro Dice: Baseline `0.7556`, Candidate `0.8219` (+0.0664)
  - False Positive Pixels: `1,449,897` down to `297,985` (-79.4%)

---

## 4. End-to-End Post-Release Integration & Rollback Test Suite

Execute the multi-scene integration suite across oil, clean ocean, and look-alike scenes:

```bash
python ml/scripts/post_release_integration_test.py
```

Audits Executed:
1. Cryptographic hash verification of production and backup files
2. Controlled sandbox rollback drill
3. Multi-scene end-to-end inference (00492.tif, clean ocean baseline, low-wind dark pool)
4. Modality discrimination (rejection of optical RGB, annotated overlays, corrupt files)
5. Classifier download process inspection

---

## 5. Backend FastAPI App Verification

Verify the backend service loads the promoted ONNX models and runs without exceptions:

```bash
python backend/test_backend.py
```

---

## 6. PyTorch vs. ONNX Runtime Parity Verification

Measure numerical concordance between PyTorch checkpoints and ONNX production exports:

```bash
python -c "
import torch, onnxruntime as ort, numpy as np
from ml.models import get_segmenter, get_classifier

# Segmenter parity (512x512)
seg_pt = get_segmenter(in_channels=2)
st = torch.load('ml/checkpoints/best_segmenter_v3_exp1_lr1e4.pt', map_location='cpu', weights_only=False)
seg_pt.load_state_dict(st['model_state'])
seg_pt.eval()
seg_ort = ort.InferenceSession('frontend/public/models/oil_segmenter.onnx', providers=['CPUExecutionProvider'])
dummy_seg = np.random.randn(1, 2, 512, 512).astype(np.float32)
with torch.no_grad():
    pt_out = torch.sigmoid(seg_pt(torch.from_numpy(dummy_seg))).numpy()
ort_out = torch.sigmoid(torch.from_numpy(seg_ort.run(None, {'input': dummy_seg})[0])).numpy()
print(f'Segmenter Max Diff: {np.max(np.abs(pt_out - ort_out)):.2e}, Concordance: {np.mean((pt_out>=0.5)==(ort_out>=0.5))*100:.2f}%')

# Classifier parity (400x400)
cls_pt = get_classifier(in_channels=2)
st_c = torch.load('ml/checkpoints/best_dualpol_classifier.pt', map_location='cpu', weights_only=False)
cls_pt.load_state_dict(st_c['model_state'])
cls_pt.eval()
cls_ort = ort.InferenceSession('frontend/public/models/oil_classifier.onnx', providers=['CPUExecutionProvider'])
dummy_cls = np.random.randn(1, 2, 400, 400).astype(np.float32)
with torch.no_grad():
    pt_c = torch.softmax(cls_pt(torch.from_numpy(dummy_cls)), dim=-1).numpy()
ort_c = torch.softmax(torch.from_numpy(cls_ort.run(None, {'input': dummy_cls})[0]), dim=-1).numpy()
print(f'Classifier Max Diff: {np.max(np.abs(pt_c - ort_c)):.2e}, Concordance: {np.mean(np.argmax(pt_c,-1)==np.argmax(ort_c,-1))*100:.2f}%')
"
```

---

## 7. Paired Scene Bootstrap 95% Confidence Intervals

Compute inferential paired bootstrap statistics across the 20 held-out test scenes:

```bash
python -c "
import pandas as pd, numpy as np
df = pd.read_csv('ml/reports/final_model_audit/per_scene_segmentation_metrics.csv')
n = len(df)
d_dice = df['candidate_dice'] - df['baseline_dice']
d_iou = df['candidate_iou'] - df['baseline_iou']
np.random.seed(42)
b_dice, b_iou = [], []
for _ in range(10000):
    idx = np.random.choice(n, size=n, replace=True)
    b_dice.append(d_dice.iloc[idx].mean())
    b_iou.append(d_iou.iloc[idx].mean())
print(f'Paired Dice Delta: {d_dice.mean():.4f}, 95% CI: [{np.percentile(b_dice, 2.5):.4f}, {np.percentile(b_dice, 97.5):.4f}]')
print(f'Paired IoU Delta:  {d_iou.mean():.4f}, 95% CI: [{np.percentile(b_iou, 2.5):.4f}, {np.percentile(b_iou, 97.5):.4f}]')
"
```

---

## 8. Resumable Classifier Independent Test Downloader & Evaluator

Check progress of the active Part III download without launching duplicate workers:

```bash
python -c "
import json
with open('ml/logs/part3_download_status.json') as f:
    st = json.load(f)
print(f'Phase: {st[\"phase\"]}, Downloaded: {st[\"bytes_downloaded\"] / 1e9:.2f} GB / {st[\"total_bytes\"] / 1e9:.2f} GB ({st[\"percent\"]}%), ETA: {st[\"eta_hours\"]} hrs')
"
```

Run independent classifier evaluation once extraction is complete:

```bash
python ml/scripts/evaluate_part3_independent.py
```
