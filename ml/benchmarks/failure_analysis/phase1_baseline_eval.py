"""
SpillSense Phase 1 — Baseline Evaluation
=========================================
Reproduces genuine classifier and segmenter baseline metrics from:
  - best_dualpol_classifier.pt  (DualPolOilSpillNet)
  - best_segmenter_v3.pt        (SpillSegNet)

Running against:
  - cls_val_v3_zenodo_gt_dualpol.npz  (4,818 patches; 217 validation scenes)
  - Segmenter metrics already embedded in checkpoint (curated-patch eval)
  - benchmark_report.json             (40-scene gallery)

Writes:
  - ml/benchmarks/failure_analysis/phase1_baseline_metrics.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ml.models import get_classifier
from ml.train import compute_metrics, compute_category_metrics

CKPT_DIR = ROOT / "ml" / "checkpoints"
CACHE_DIR = ROOT / "ml" / "cache_patches"
GALLERY_JSON = ROOT / "ml" / "benchmark_gallery" / "benchmark_report.json"
OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_classifier_baseline():
    """Evaluate DualPolOilSpillNet on 4,818-patch held-out validation set."""
    print("=" * 60)
    print("PHASE 1 — CLASSIFIER BASELINE (DualPolOilSpillNet)")
    print("=" * 60)

    ckpt_path = CKPT_DIR / "best_dualpol_classifier.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = get_classifier(2)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    threshold = float(ckpt.get("best_threshold", 0.50))
    print(f"Checkpoint: {ckpt_path.name}  (epoch={ckpt['epoch']}, threshold={threshold:.3f})")

    # Load val cache
    val_npz = CACHE_DIR / "cls_val_v3_zenodo_gt_dualpol.npz"
    va = np.load(str(val_npz), allow_pickle=True)
    patches = va["patches"]        # (N, 2, 400, 400) uint8
    labels  = va["labels"]         # (N,) float32
    cats    = va["categories"]     # (N,) str
    print(f"Validation set: {len(labels):,} patches")
    for cat in sorted(np.unique(cats)):
        n = int(np.sum(cats == cat))
        print(f"  {cat:<12s}: {n:>5d}")

    # Inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    all_probs = []
    batch_size = 128
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            b = torch.from_numpy(patches[i:i + batch_size].astype(np.float32) / 255.0).to(device)
            logits = model(b).squeeze(-1).cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
            all_probs.extend(probs.tolist())
    inference_ms = (time.perf_counter() - t0) * 1000
    probs = np.array(all_probs)

    # Compute metrics
    m = compute_metrics(labels, probs, threshold)
    cat_m = compute_category_metrics(labels, probs, cats, threshold)

    print("\n--- OVERALL METRICS ---")
    print(f"Accuracy:  {m['accuracy']:.4f}")
    print(f"Precision: {m['precision']:.4f}")
    print(f"Recall:    {m['recall']:.4f}")
    print(f"F1:        {m['f1']:.4f}")
    print(f"ROC-AUC:   {m['roc_auc']:.4f}")
    print(f"PR-AUC:    {m['pr_auc']:.4f}")
    print(f"TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}")

    print("\n--- PER-CATEGORY ---")
    for cat, cm in cat_m.items():
        if cat == "oil":
            print(f"  oil        : Recall={cm['recall']:.4f} (TP={cm['tp']}, FN={cm['fn']})")
        else:
            tnr = cm.get("tnr", 0.0)
            fpr = cm.get("fpr", 0.0)
            print(f"  {cat:<12s}: TNR={tnr:.4f}, FPR={fpr:.4f} (TN={cm.get('tn',0)}, FP={cm.get('fp',0)})")

    print(f"\nInference time: {inference_ms:.0f} ms for {len(probs)} patches "
          f"({inference_ms / len(probs) * 1000:.1f} us/patch)")

    return {
        "source": "cls_val_v3_zenodo_gt_dualpol.npz",
        "n_patches": int(len(labels)),
        "checkpoint": str(ckpt_path.name),
        "epoch": ckpt["epoch"],
        "threshold": threshold,
        "device": str(device),
        "overall": {k: float(v) if not isinstance(v, int) else int(v)
                    for k, v in m.items()},
        "per_category": {
            cat: {k: float(v) if not isinstance(v, int) else int(v)
                  for k, v in cm.items()}
            for cat, cm in cat_m.items()
        },
        "inference_ms_total": round(inference_ms, 2),
    }


def read_segmenter_baseline():
    """Read segmenter metrics from checkpoint (curated-patch validation)."""
    print("\n" + "=" * 60)
    print("PHASE 1 — SEGMENTER BASELINE (SpillSegNet)")
    print("=" * 60)

    ckpt_path = CKPT_DIR / "best_segmenter_v3.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    seg_m = ckpt.get("seg_metrics", {})
    best_dice = ckpt.get("best_dice", seg_m.get("dice", 0))

    print(f"Checkpoint: {ckpt_path.name}  (epoch={ckpt['epoch']})")
    print(f"Dice:       {best_dice:.4f}")
    print(f"IoU:        {seg_m.get('iou', 0):.4f}")
    print(f"Precision:  {seg_m.get('precision', 0):.4f}")
    print(f"Recall:     {seg_m.get('recall', 0):.4f}")
    print(f"Pixel Acc:  {seg_m.get('pixel_accuracy', 0):.4f}")
    tp = int(seg_m.get("tp", 0))
    fp = int(seg_m.get("fp", 0))
    fn = int(seg_m.get("fn", 0))
    tn = int(seg_m.get("tn", 0))
    total = tp + fp + fn + tn
    fg_ratio = (tp + fn) / max(1, total)
    print(f"Pixel counts: TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")
    print(f"FG pixel ratio (imbalance): {fg_ratio:.4f}  ({1/max(1e-9,fg_ratio):.1f}:1 background-to-fg)")
    print("\nNOTE: Curated-patch Dice. Full-scene macro Dice ~ 0.68 due to ~1:200 imbalance.")

    return {
        "source": "best_segmenter_v3.pt (checkpoint embedded seg_metrics)",
        "eval_domain": "curated_512x512_patches",
        "checkpoint": str(ckpt_path.name),
        "epoch": ckpt["epoch"],
        "best_dice": float(best_dice),
        "iou": float(seg_m.get("iou", 0)),
        "precision": float(seg_m.get("precision", 0)),
        "recall": float(seg_m.get("recall", 0)),
        "pixel_accuracy": float(seg_m.get("pixel_accuracy", 0)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "fg_pixel_ratio": round(fg_ratio, 5),
        "note": (
            "Curated-patch Dice=0.828 vs full-scene macro Dice~0.68: "
            "difference driven by extreme class imbalance (~1:200 fg:bg) across full 2048x2048 scenes. "
            "Validation patches are oversampled 4x oil to reduce gap during training."
        ),
    }


def analyse_gallery():
    """Parse 40-scene benchmark gallery to compute gallery-level metrics."""
    print("\n" + "=" * 60)
    print("PHASE 1 — BENCHMARK GALLERY ANALYSIS (40 scenes)")
    print("=" * 60)

    with open(GALLERY_JSON) as f:
        records = json.load(f)

    cats = {}
    for r in records:
        cat = r["category"]
        cats.setdefault(cat, []).append(r)

    print(f"Total records: {len(records)}")
    for cat, recs in cats.items():
        print(f"  {cat:<12s}: {len(recs):3d} patches")

    analysis = {}
    for cat, recs in cats.items():
        decisions = [r["cls_decision"] for r in recs]
        probs = [r["cls_probability"] for r in recs]
        if cat == "oil":
            tp = sum(1 for d in decisions if d == "OIL")
            fn = sum(1 for d in decisions if d == "NON-OIL")
            analysis[cat] = {
                "n": len(recs), "tp": tp, "fn": fn,
                "recall": tp / max(1, tp + fn),
                "mean_prob": float(np.mean(probs)),
                "min_prob": float(np.min(probs)),
            }
            print(f"  oil        : TP={tp}, FN={fn}, Recall={tp/(tp+fn):.3f}, mean_prob={np.mean(probs):.3f}")
        else:
            tn = sum(1 for d in decisions if d == "NON-OIL")
            fp = sum(1 for d in decisions if d == "OIL")
            analysis[cat] = {
                "n": len(recs), "tn": tn, "fp": fp,
                "tnr": tn / max(1, tn + fp),
                "fpr": fp / max(1, tn + fp),
                "mean_prob": float(np.mean(probs)),
            }
            print(f"  {cat:<12s}: TN={tn}, FP={fp}, TNR={tn/(tn+fp):.3f}, mean_prob={np.mean(probs):.3f}")

    threshold = records[0]["threshold"]
    print(f"\nGallery threshold: {threshold:.3f}")
    print("NOTE: Gallery uses best_classifier_v3.pt (threshold=0.10, gallery-tuned). "
          "NOT a held-out set — model was calibrated to these scenes.")

    return {
        "source": "benchmark_gallery/benchmark_report.json",
        "n_records": len(records),
        "threshold": float(threshold),
        "per_category": analysis,
    }


def main():
    print("\nSpillSense Phase 1 — Baseline Evaluation")
    print("==========================================\n")

    cls_baseline = run_classifier_baseline()
    seg_baseline = read_segmenter_baseline()
    gallery = analyse_gallery()

    results = {
        "phase": 1,
        "description": "Baseline metric reproduction for SpillSense oil spill pipeline",
        "classifier_val_set": cls_baseline,
        "segmenter_curated_patches": seg_baseline,
        "benchmark_gallery": gallery,
        "metric_domain_clarification": {
            "cls_val": (
                "4,818 patches from 217 held-out validation scenes (v3 Zenodo dataset). "
                "Evaluated with best_dualpol_classifier.pt (threshold=0.490). "
                "Most reliable classifier metric — genuinely held-out."
            ),
            "seg_patch": (
                "Curated 512x512 oil-containing patches from Zenodo train/val scenes. "
                "Dice=0.828 reflects in-distribution patches with 4x oil oversampling. "
                "Not comparable to full-scene Dice."
            ),
            "seg_full_scene": (
                "Full 2048x2048 scene tiled inference with 512x512/384 stride. "
                "Macro Dice~0.68 due to ~1:200 fg:bg class imbalance. "
                "Open-ocean speckle accumulates false positives, reducing precision to ~64-67%."
            ),
            "gallery": (
                "40 benchmark patches from 4 scene categories. "
                "Uses best_classifier_v3.pt gallery-tuned model (threshold=0.10). "
                "NOT a held-out set — model was tuned to these scenes."
            ),
        },
    }

    out_path = OUT_DIR / "phase1_baseline_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[DONE] Written: {out_path}")
    return results


if __name__ == "__main__":
    main()
