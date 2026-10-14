# SpillSense — Phase 0 Baseline Backup Record

## Git State

| Item | Value |
|------|-------|
| Backup branch | `improvement/phase0-backup` |
| Origin branch | `main` |
| HEAD commit | `53e67ee` — fix(ml): eliminate SAR Detection Lab false positive on clean ocean and retrain classifier for 35 epochs |
| Rollback command | `git checkout main` |

## Checkpoint SHA-256 Hashes (unmodified originals)

| File | SHA-256 |
|------|---------|
| best_classifier_v3.pt | `076A5AA4224FA90050EB29526CD9AD8772D4CF38257FCCF56B43AD09A4930705` |
| best_classifier_v3_finetuned.pt | `DA224DE8D0C41CC39D7E1C8606658D16AB786334A9307D96715E524D16A88F6D` |
| best_dualpol_classifier.pt | `262EF1A8D1F3FBCFBBEDBCC67AEE0C3C1FAF5995CC855CF7131D1FD14C78D902` |
| best_segmenter_v3.pt | `9C5901016C19CB93B8B492512936D8726E1B988F32AF8E392A5225C260C08B08` |

## Verified Baseline Metrics (from checkpoints)

### Classifier: best_dualpol_classifier.pt
- Epoch: 7
- Val Accuracy: 95.52%
- Val F1: 0.890
- Val ROC-AUC: 0.981
- Val PR-AUC: 0.523
- Threshold: 0.490
- Val set: 4,022 patches (at time of training; 4,818 at current cache)

### Segmenter: best_segmenter_v3.pt
- Epoch: 2  ← NOTE: Training stopped extremely early
- Curated-patch Dice: 0.8279
- Curated-patch IoU: 0.7063
- Curated-patch Precision: 0.7987
- Curated-patch Recall: 0.8593
- Pixel counts: TP=1,844,039 FP=464,660 FN=302,043 TN=13,642,186

### Classifier Gallery: best_classifier_v3.pt
- Epoch: 35
- Gallery threshold: 0.10 (tuned to gallery — NOT a held-out metric)
- Gallery best_score: 1.0 (over-fit to 40 gallery scenes)

## Recovery Procedure

1. Verify hashes still match: `Get-FileHash ml/checkpoints/*.pt -Algorithm SHA256`
2. If any checkpoint is corrupted, restore from Git LFS or download Zenodo dataset and retrain.
3. To revert all code changes: `git checkout main`
4. To revert only specific files: `git checkout main -- ml/train_segmentation.py`

## What Will Be Modified in Phase 3

Only NEW checkpoint files will be created (e.g., `best_segmenter_v3_exp1_lr1e4.pt`).
The original checkpoints listed above will never be overwritten.
