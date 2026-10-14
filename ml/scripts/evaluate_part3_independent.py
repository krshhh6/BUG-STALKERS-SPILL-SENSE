"""
SpillSense Phase 4 — Independent Classifier Test Suite (Zenodo Part III)
========================================================================
Evaluates frozen baseline (best_dualpol_classifier.pt, threshold 0.49) vs.
frozen candidate (best_dualpol_classifier_exp2_sampling_sw30.pt, threshold 0.43)
on the official 450-scene independent Zenodo Part III test set.

Acceptance Criteria:
  1. Oil Recall >= 90.0%
  2. Ship-Wake FPR <= 5.0%
  3. Overall F1 >= 0.950

Zero retraining. Zero threshold tuning on test data.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.config import CONFIG
from ml.models import get_classifier
from ml.preprocess import load_sentinel1_scene, load_mask, normalize_sar_dualpol
from ml.train import compute_metrics, compute_category_metrics


def find_test_scenes(test_root: Path = Path("D:/Test")) -> Dict[str, List[Dict]]:
    """Auto-discover extracted test scenes across D:/Test."""
    scenes = {"oil": [], "clean": [], "lookalike": []}

    # Search for extracted image directories
    all_tifs = list(test_root.rglob("*.tif"))
    print(f"[TEST DATASET] Found {len(all_tifs)} TIFF files in {test_root}")

    for tif in all_tifs:
        p_str = str(tif).lower()
        if "mask" in p_str:
            continue

        # Look for corresponding mask
        mask_candidate = Path(str(tif).replace("Images", "Mask").replace("images", "mask"))
        has_mask = mask_candidate.exists()

        if "oil" in p_str and "no" not in p_str and "look" not in p_str:
            scenes["oil"].append({"image_path": str(tif), "mask_path": str(mask_candidate) if has_mask else "", "has_mask": has_mask})
        elif "no_oil" in p_str or "nooil" in p_str or "clean" in p_str:
            scenes["clean"].append({"image_path": str(tif), "mask_path": str(mask_candidate) if has_mask else "", "has_mask": has_mask})
        elif "lookalike" in p_str or "look_alike" in p_str:
            scenes["lookalike"].append({"image_path": str(tif), "mask_path": str(mask_candidate) if has_mask else "", "has_mask": has_mask})

    print(f"  Discovered Oil Scenes:       {len(scenes['oil'])}")
    print(f"  Discovered Clean Scenes:     {len(scenes['clean'])}")
    print(f"  Discovered Look-alike Scenes: {len(scenes['lookalike'])}")
    return scenes


def evaluate_model_on_patches(model: nn.Module, patches: np.ndarray, labels: np.ndarray,
                              categories: np.ndarray, threshold: float,
                              batch_size: int = 64, device: torch.device = None) -> Dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch = patches[i:i+batch_size].astype(np.float32) / 255.0
            t_batch = torch.from_numpy(batch).to(device)
            logits = model(t_batch)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            all_probs.extend(probs)

    all_probs = np.array(all_probs)
    preds = (all_probs >= threshold).astype(float)

    base_metrics = compute_metrics(preds, labels, all_probs)
    cat_metrics = compute_category_metrics(preds, labels, categories)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "total_examples": int(len(labels)),
        "accuracy": float(base_metrics["accuracy"]),
        "precision": float(base_metrics["precision"]),
        "recall": float(base_metrics["recall"]),
        "f1": float(base_metrics["f1"]),
        "roc_auc": float(base_metrics["roc_auc"]),
        "pr_auc": float(base_metrics["pr_auc"]),
        "confusion_matrix": {
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)
        },
        "category_metrics": cat_metrics,
        "clean_fpr": float(cat_metrics.get("clean", {}).get("fpr", 0.0)),
        "lookalike_fpr": float(cat_metrics.get("lookalike", {}).get("fpr", 0.0)),
        "ship_wake_fpr": float(cat_metrics.get("ship_wake", {}).get("fpr", 0.0)),
    }


def run_independent_test():
    print("=" * 80)
    print("SPILLSENSE PHASE 4 — CLASSIFIER INDEPENDENT TEST EVALUATION")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Discover Part III scenes
    test_scenes = find_test_scenes()
    total_found = sum(len(v) for v in test_scenes.values())
    if total_found < 50:
        print(f"\n[BLOCKED] Only {total_found} scenes found in D:/Test. Archive download or extraction is still in progress.")
        print("  Current download status can be monitored via ml/logs/part3_download_status.json")
        return False

    # 2. Load Checkpoints
    ckpt_dir = ROOT / "ml" / "checkpoints"
    base_ckpt_p = ckpt_dir / "best_dualpol_classifier.pt"
    cand_ckpt_p = ckpt_dir / "best_dualpol_classifier_exp2_sampling_sw30.pt"

    base_ckpt = torch.load(base_ckpt_p, map_location=device, weights_only=False)
    cand_ckpt = torch.load(cand_ckpt_p, map_location=device, weights_only=False)

    model_base = get_classifier(2).to(device)
    model_base.load_state_dict(base_ckpt["model_state"])
    model_base.eval()

    model_cand = get_classifier(2).to(device)
    model_cand.load_state_dict(cand_ckpt["model_state"])
    model_cand.eval()

    # 3. Patch Extraction from Test Scenes (Placeholder / loader for real extracted test NPZ)
    test_cache = ROOT / "ml" / "cache_patches" / "cls_test_part3_dualpol.npz"
    if not test_cache.exists():
        print(f"[CACHE MISS] Need to build test patch cache from {total_found} scenes...")
        return False

    data = np.load(test_cache, allow_pickle=True)
    patches = data["patches"]
    labels = data["labels"]
    cats = data["categories"]

    print(f"\n[EVALUATION] Evaluating frozen models on {len(labels):,} test patches...")
    res_base = evaluate_model_on_patches(model_base, patches, labels, cats, threshold=0.49, device=device)
    res_cand = evaluate_model_on_patches(model_cand, patches, labels, cats, threshold=0.43, device=device)

    print("\nRESULTS SUMMARY (FROZEN EVALUATION — NO TUNING):")
    print(f"Baseline (Thresh 0.49): Recall={res_base['recall']:.4f}, Precision={res_base['precision']:.4f}, F1={res_base['f1']:.4f}, Wake FPR={res_base['ship_wake_fpr']:.4f}")
    print(f"Candidate (Thresh 0.43): Recall={res_cand['recall']:.4f}, Precision={res_cand['precision']:.4f}, F1={res_cand['f1']:.4f}, Wake FPR={res_cand['ship_wake_fpr']:.4f}")

    # Acceptance Criteria Check
    c1 = res_cand["recall"] >= 0.900
    c2 = res_cand["ship_wake_fpr"] <= 0.050
    c3 = res_cand["f1"] >= 0.950
    promoted = c1 and c2 and c3

    print(f"\nAcceptance Criteria Check:")
    print(f"  1. Oil Recall >= 90%:      {res_cand['recall']*100:.2f}% -> {'PASS' if c1 else 'FAIL'}")
    print(f"  2. Ship-Wake FPR <= 5%:    {res_cand['ship_wake_fpr']*100:.2f}% -> {'PASS' if c2 else 'FAIL'}")
    print(f"  3. Overall F1 >= 0.95:     {res_cand['f1']:.4f} -> {'PASS' if c3 else 'FAIL'}")
    print(f"  --> Promotion Decision:    {'PROMOTE' if promoted else 'HOLD IN STAGING'}")

    return promoted


if __name__ == "__main__":
    run_independent_test()
