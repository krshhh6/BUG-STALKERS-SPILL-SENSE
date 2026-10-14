"""Quick threshold analysis for dualpol classifier on val cache."""
import numpy as np
import torch
import sys
sys.path.insert(0, 'd:/Spill Sense/SIH-26143-OIL-Spill')
from ml.models import get_classifier
from ml.train import compute_metrics

ckpt = torch.load('d:/Spill Sense/SIH-26143-OIL-Spill/ml/checkpoints/best_dualpol_classifier.pt', map_location='cpu', weights_only=False)
model = get_classifier(2)
model.load_state_dict(ckpt['model_state'])
model.eval()

va = np.load('d:/Spill Sense/SIH-26143-OIL-Spill/ml/cache_patches/cls_val_v3_zenodo_gt_dualpol.npz', allow_pickle=True)
patches, labels, cats = va['patches'], va['labels'], va['categories']

all_probs = []
with torch.no_grad():
    for i in range(0, len(patches), 128):
        b = torch.from_numpy(patches[i:i+128].astype(np.float32) / 255.0)
        logits = model(b).squeeze(-1).numpy()
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
        all_probs.extend(probs.tolist())
probs = np.array(all_probs)

print('=== PROBABILITY DISTRIBUTION ===')
print(f'  Mean: {probs.mean():.4f}')
print(f'  Std:  {probs.std():.4f}')
print(f'  Median: {np.median(probs):.4f}')
print(f'  P25: {np.percentile(probs, 25):.4f}')
print(f'  P75: {np.percentile(probs, 75):.4f}')

oil_probs = probs[cats == 'oil']
non_oil_probs = probs[cats != 'oil']
print()
print(f'OIL patches (n={len(oil_probs)}):')
print(f'  Mean={oil_probs.mean():.4f}, Median={np.median(oil_probs):.4f}')
print(f'  Fraction above 0.49: {(oil_probs >= 0.49).mean():.4f}')
print(f'  Fraction above 0.30: {(oil_probs >= 0.30).mean():.4f}')
print(f'  Fraction above 0.20: {(oil_probs >= 0.20).mean():.4f}')
print()
print(f'NON-OIL patches (n={len(non_oil_probs)}):')
print(f'  Mean={non_oil_probs.mean():.4f}, Fraction above 0.49: {(non_oil_probs >= 0.49).mean():.4f}')

print()
print('=== THRESHOLD SWEEP ===')
thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.27, 0.30, 0.35, 0.40, 0.49, 0.50, 0.60]
for t in thresholds:
    m = compute_metrics(labels, probs, t)
    acc = m['accuracy']
    pre = m['precision']
    rec = m['recall']
    f1  = m['f1']
    tn, fp, fn, tp = m['tn'], m['fp'], m['fn'], m['tp']
    print(f't={t:.2f}: Acc={acc:.3f} P={pre:.3f} R={rec:.3f} F1={f1:.3f} | TN={tn} FP={fp} FN={fn} TP={tp}')
