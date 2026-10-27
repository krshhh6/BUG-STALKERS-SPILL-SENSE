"""
SpillSense External Validation - Benchmark Pipeline
=====================================================
Runs frozen production models on external datasets and computes metrics.
No retraining, no threshold tuning on test set, no fabricated results.
"""

import os
import sys
import csv
import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from PIL import Image

from ml.config import CONFIG, SpillSenseConfig
from ml.models import get_segmenter, get_classifier, SpillSegNet, DualPolOilSpillNet, count_params
from ml.preprocess import normalize_sar_dualpol

BENCHMARK_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BENCHMARK_DIR / "datasets"
FIGURES_DIR = BENCHMARK_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================================
# Model Loading
# ======================================================================

def load_production_segmenter(device=None):
    """Load the production segmenter checkpoint (frozen)."""
    if device is None:
        device = CONFIG.get_device()
    ckpt_path = CONFIG.checkpoint_dir / "best_segmenter_v3_exp1_lr1e4.pt"
    if not ckpt_path.exists():
        ckpt_path = CONFIG.checkpoint_dir / "best_segmenter_v3.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No segmenter checkpoint found in {CONFIG.checkpoint_dir}")

    model = get_segmenter(in_channels=2)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)

    seg_metrics = ckpt.get("seg_metrics", {})
    info = {
        "checkpoint": ckpt_path.name,
        "epoch": ckpt.get("epoch", "N/A"),
        "params": count_params(model),
        "threshold": 0.50,
        "reference_dice": seg_metrics.get("dice", ckpt.get("best_dice", None)),
        "reference_iou": seg_metrics.get("iou", None),
    }
    print(f"[SEGMENTER] Loaded {ckpt_path.name} ({info['params']:,} params)")
    return model, info


def load_production_classifier(device=None):
    """Load the production classifier checkpoint (frozen)."""
    if device is None:
        device = CONFIG.get_device()
    ckpt_path = CONFIG.checkpoint_dir / "best_dualpol_classifier.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Classifier not found: {ckpt_path}")

    model = get_classifier(in_channels=2)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)

    info = {
        "checkpoint": ckpt_path.name,
        "params": count_params(model),
        "threshold": ckpt.get("best_threshold", 0.49),
    }
    print(f"[CLASSIFIER] Loaded {ckpt_path.name} ({info['params']:,} params, thr={info['threshold']:.2f})")
    return model, info


def load_staged_classifier(device=None):
    """Load the staged candidate classifier (frozen)."""
    if device is None:
        device = CONFIG.get_device()
    ckpt_path = CONFIG.checkpoint_dir / "best_dualpol_classifier_exp2_sampling_sw30.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Staged classifier not found: {ckpt_path}")

    model = get_classifier(in_channels=2)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)

    info = {
        "checkpoint": ckpt_path.name,
        "params": count_params(model),
        "threshold": ckpt.get("best_threshold", 0.43),
    }
    print(f"[STAGED CLS] Loaded {ckpt_path.name} (thr={info['threshold']:.2f})")
    return model, info


# ======================================================================
# Metric Computation
# ======================================================================

def compute_seg_metrics(pred_probs, gt_mask, threshold=0.5):
    """Compute segmentation metrics from probability map and binary mask."""
    pred_bin = (pred_probs >= threshold).astype(np.int32)
    gt_bin = (gt_mask > 0).astype(np.int32)

    tp = int(np.sum((pred_bin == 1) & (gt_bin == 1)))
    fp = int(np.sum((pred_bin == 1) & (gt_bin == 0)))
    fn = int(np.sum((pred_bin == 0) & (gt_bin == 1)))
    tn = int(np.sum((pred_bin == 0) & (gt_bin == 0)))

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    dice = 2 * tp / max(1, 2 * tp + fp + fn)
    iou = tp / max(1, tp + fp + fn)
    fpr = fp / max(1, fp + tn)
    fnr = fn / max(1, fn + tp)
    pixel_acc = (tp + tn) / max(1, tp + tn + fp + fn)

    return {
        "dice": dice, "iou": iou,
        "precision": precision, "recall": recall,
        "f1": dice,  # F1 == Dice for binary
        "fpr": fpr, "fnr": fnr,
        "pixel_accuracy": pixel_acc,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "oil_pixels_gt": int(np.sum(gt_bin)),
        "total_pixels": int(gt_bin.size),
    }


def compute_cls_metrics(probs, labels, threshold=0.5):
    """Compute classification metrics."""
    preds = (np.array(probs) >= threshold).astype(int)
    labels = np.array(labels).astype(int)

    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    specificity = tn / max(1, tn + fp)
    fpr = fp / max(1, fp + tn)

    result = {
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1,
        "specificity": specificity, "fpr": fpr,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }

    # ROC-AUC and PR-AUC
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        if len(set(labels)) > 1:
            result["roc_auc"] = float(roc_auc_score(labels, probs))
            result["pr_auc"] = float(average_precision_score(labels, probs))
    except Exception:
        pass

    return result


# ======================================================================
# Dataset-Specific Evaluators
# ======================================================================

def evaluate_sos_dataset(seg_model, seg_info, device):
    """
    Evaluate segmenter on Refined Deep-SAR Oil Spill (SOS) dataset.
    SOS contains single-polarization SAR patches with binary masks.
    Our model expects dual-pol (VV+VH) -- compatibility must be checked.
    """
    print("\n" + "=" * 60)
    print("DATASET A: Refined Deep-SAR Oil Spill (SOS)")
    print("=" * 60)

    sos_dir = DATASETS_DIR / "A_SOS"
    images_dir = None
    masks_dir = None

    # Find extracted directories (prioritize val split)
    for candidate in [sos_dir / "images" / "val", sos_dir / "images" / "test",
                      sos_dir / "images", sos_dir / "val" / "images"]:
        if candidate.exists() and any(candidate.glob("*.png")):
            images_dir = candidate
            break

    if images_dir is None:
        for candidate in [sos_dir / "images", sos_dir]:
            if candidate.exists():
                cands = list(candidate.glob("*/*.png"))
                if cands:
                    images_dir = cands[0].parent
                    break

    # Find masks
    for candidate in [sos_dir / "masks" / "val", sos_dir / "masks" / "test",
                      sos_dir / "masks", sos_dir / "val" / "masks"]:
        if candidate.exists() and any(candidate.glob("*.png")):
            masks_dir = candidate
            break

    if masks_dir is None and images_dir is not None:
        cand_masks = sos_dir / "masks" / images_dir.name
        if cand_masks.exists():
            masks_dir = cand_masks

    # Inspect dataset format
    img_files = sorted(list(images_dir.rglob("*.jpg")) + list(images_dir.rglob("*.png")) + list(images_dir.rglob("*.tif")))
    print(f"  Found {len(img_files)} image files")

    if not img_files:
        return None

    # Check a sample image
    sample = np.array(Image.open(str(img_files[0])))
    print(f"  Sample image shape: {sample.shape}, dtype: {sample.dtype}")
    print(f"  Value range: [{sample.min()}, {sample.max()}]")

    # COMPATIBILITY CHECK
    # SOS is single-polarization (1-channel grayscale or 3-channel RGB-rendered SAR)
    # Our model requires 2-channel (VV, VH) dual-pol input
    is_single_pol = sample.ndim == 2 or (sample.ndim == 3 and sample.shape[-1] in (1, 3))

    if is_single_pol:
        print("\n  [COMPATIBILITY] SOS is single-polarization.")
        print("  Our segmenter requires dual-pol (VV + VH) input.")
        print("  Running ADAPTED evaluation: duplicating single channel as VV,")
        print("  synthesizing VH with -10.5 dB offset (physical cross-pol approximation).")
        print("  THIS IS NOT A DIRECT ZERO-SHOT EVALUATION.")

    results = []
    seg_threshold = seg_info["threshold"]

    for idx, img_path in enumerate(img_files):
        try:
            img = np.array(Image.open(str(img_path)).convert("L")).astype(np.float32)

            # Find matching mask
            stem = img_path.stem
            mask_path = None
            for ext in [".png", ".jpg", ".tif", ".bmp"]:
                candidate = masks_dir / (stem + ext)
                if candidate.exists():
                    mask_path = candidate
                    break
            if mask_path is None:
                # Try matching by pattern
                for mp in masks_dir.rglob(f"*{stem}*"):
                    mask_path = mp
                    break

            if mask_path is None:
                continue

            mask = np.array(Image.open(str(mask_path)).convert("L")).astype(np.float32)
            mask = (mask > 127).astype(np.uint8)  # Binarize

            h, w = img.shape

            # Normalize to approximate dB range for our model
            # SOS images are typically 0-255 scaled SAR intensity
            # Map to approximate Sigma0 dB range
            img_norm = img / 255.0

            # For our dual-pol model: create 2-channel input
            # Channel 0 = VV (normalized), Channel 1 = VH (offset)
            vv_norm = img_norm
            vh_norm = np.clip(img_norm - 0.20, 0.0, 1.0)  # Approximate cross-pol offset
            input_2ch = np.stack([vv_norm, vh_norm], axis=0).astype(np.float32)

            # Resize to 512x512 for segmenter
            input_tensor = torch.from_numpy(input_2ch).unsqueeze(0).float()
            input_tensor = F.interpolate(input_tensor, size=(512, 512), mode="bilinear", align_corners=False)
            input_tensor = input_tensor.to(device)

            with torch.no_grad():
                logits = seg_model(input_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()[0, 0]

            # Resize prediction back to original dimensions
            prob_resized = np.array(Image.fromarray((probs * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255.0

            # Compute metrics
            m = compute_seg_metrics(prob_resized, mask, threshold=seg_threshold)
            m["scene_id"] = stem
            m["dataset"] = "A_SOS"
            results.append(m)

            if idx < 5 or (idx + 1) % 50 == 0:
                print(f"    [{idx+1}/{len(img_files)}] {stem}: Dice={m['dice']:.4f} IoU={m['iou']:.4f}")

            # Save visual for first few
            if idx < 3:
                save_visual(img_path, mask, prob_resized, seg_threshold,
                           FIGURES_DIR / f"sos_{stem}_overlay.png")

        except Exception as e:
            print(f"    [WARN] Failed on {img_path.name}: {e}")
            continue

    if results:
        agg = aggregate_seg_results(results)
        agg["dataset"] = "A_SOS"
        agg["evaluation_type"] = "adapted (single-pol to dual-pol synthesis)"
        agg["num_scenes"] = len(results)
        print(f"\n  SOS Aggregate: Dice={agg['micro_dice']:.4f} IoU={agg['micro_iou']:.4f} "
              f"Prec={agg['micro_precision']:.4f} Rec={agg['micro_recall']:.4f}")
        return {"per_scene": results, "aggregate": agg}

    return None


def evaluate_cantarell_dataset(seg_model, seg_info, device):
    """
    Evaluate segmenter on Cantarell oil-spill/seepage dataset.
    173 image-mask triplets, ~800x600 px, RGB-rendered SAR images.
    Evaluates both:
      1. test_split (22 tiles) -- direct comparison with published TinyUNet, Hybrid, and Classical
      2. full_dataset (173 tiles) -- direct comparison with published full-dataset Classical
    """
    print("\n" + "=" * 60)
    print("DATASET D: Cantarell Oil Spill/Seepage")
    print("=" * 60)

    cant_dir = DATASETS_DIR / "D_Cantarell"
    repo_dir = None
    for candidate in [cant_dir / "repo",
                      cant_dir / "oilspill-detection-dataset-main",
                      cant_dir / "oilspill-detection-dataset"]:
        if candidate.exists() and (candidate / "IMAGES").exists():
            repo_dir = candidate
            break

    if repo_dir is None:
        for d in cant_dir.rglob("IMAGES"):
            if d.is_dir():
                repo_dir = d.parent
                break

    if repo_dir is None:
        print("  [ERROR] Cantarell dataset IMAGES directory not found")
        return None

    images_dir = repo_dir / "IMAGES"
    splits_dir = repo_dir / "splits"
    print(f"  Repository root: {repo_dir}")

    # Read split manifests (CSV format: sample_id, image_path, mask_path)
    test_manifest = []
    full_manifest = []

    test_csv = splits_dir / "test.csv"
    if test_csv.exists():
        with open(test_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                test_manifest.append(row)
        print(f"  Loaded test manifest: {len(test_manifest)} samples")

    for split_name in ["train.csv", "val.csv", "test.csv"]:
        sp_path = splits_dir / split_name
        if sp_path.exists():
            with open(sp_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    full_manifest.append(row)
    print(f"  Loaded full manifest: {len(full_manifest)} samples across all splits")

    img_rgb_dir = images_dir / "IMG-RGB"
    labels_dir = images_dir / "LABELS-1D"

    # COMPATIBILITY
    print("\n  [COMPATIBILITY] Cantarell uses RGB-rendered SAR images (~800x600).")
    print("  Our model expects 2-channel dual-pol (VV+VH) at 512x512.")
    print("  Running ADAPTED evaluation: grayscale conversion + channel synthesis.")
    print("  THIS IS NOT A DIRECT ZERO-SHOT EVALUATION.")

    seg_threshold = seg_info["threshold"]

    def _eval_manifest(manifest, split_label):
        results = []
        for idx, row in enumerate(manifest):
            try:
                sample_id = row.get("sample_id", "")
                rel_img = row.get("image_path", "")
                rel_mask = row.get("mask_path", "")

                img_path = repo_dir / rel_img if rel_img else img_rgb_dir / f"{sample_id}.png"
                mask_path = repo_dir / rel_mask if rel_mask else labels_dir / f"{sample_id}_1D.png"

                if not img_path.exists():
                    img_cand = list(img_rgb_dir.glob(f"*{sample_id}*"))
                    if img_cand:
                        img_path = img_cand[0]

                if not mask_path.exists():
                    mask_cand = list(labels_dir.glob(f"*{sample_id}*"))
                    if mask_cand:
                        mask_path = mask_cand[0]

                if not img_path.exists() or not mask_path.exists():
                    print(f"    [WARN] Missing file for {sample_id} (img={img_path.exists()}, mask={mask_path.exists()})")
                    continue

                # Load image and convert to grayscale
                img = np.array(Image.open(str(img_path)).convert("L")).astype(np.float32)
                mask = np.array(Image.open(str(mask_path)).convert("L")).astype(np.float32)
                mask = (mask > 0).astype(np.uint8)  # Binary: 0=sea, 1=slick

                h, w = img.shape

                # Normalize and create dual-pol input
                img_norm = img / 255.0
                vv_norm = img_norm
                vh_norm = np.clip(img_norm - 0.20, 0.0, 1.0)
                input_2ch = np.stack([vv_norm, vh_norm], axis=0).astype(np.float32)

                # Resize to 512x512 for model input
                input_tensor = torch.from_numpy(input_2ch).unsqueeze(0).float()
                input_tensor = F.interpolate(input_tensor, size=(512, 512), mode="bilinear", align_corners=False)
                input_tensor = input_tensor.to(device)

                with torch.no_grad():
                    logits = seg_model(input_tensor)
                    probs = torch.sigmoid(logits).cpu().numpy()[0, 0]

                # Resize prediction back to original dimensions
                prob_resized = np.array(
                    Image.fromarray((probs * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
                ) / 255.0

                m = compute_seg_metrics(prob_resized, mask, threshold=seg_threshold)
                m["scene_id"] = sample_id
                m["dataset"] = f"D_Cantarell_{split_label}"
                results.append(m)

                if idx < 5 or (idx + 1) % 25 == 0:
                    print(f"    [{split_label} {idx+1}/{len(manifest)}] {sample_id}: Dice={m['dice']:.4f} IoU={m['iou']:.4f} Prec={m['precision']:.4f} Rec={m['recall']:.4f}")

                if split_label == "test" and idx < 3:
                    save_visual(img_path, mask, prob_resized, seg_threshold,
                               FIGURES_DIR / f"cantarell_{sample_id}_overlay.png")

            except Exception as e:
                print(f"    [WARN] Failed on {row}: {e}")
                continue
        return results

    # Evaluate test split
    test_results = _eval_manifest(test_manifest, "test") if test_manifest else []
    full_results = _eval_manifest(full_manifest, "full") if full_manifest else []

    output = {}
    if test_results:
        agg_test = aggregate_seg_results(test_results)
        agg_test["dataset"] = "D_Cantarell (test_split, 22 tiles)"
        agg_test["evaluation_type"] = "adapted (RGB-SAR to dual-pol synthesis)"
        agg_test["num_scenes"] = len(test_results)
        output["test_split"] = {"per_scene": test_results, "aggregate": agg_test}
        print(f"\n  Cantarell (Test Split, n={len(test_results)}):")
        print(f"    Micro IoU:  {agg_test['micro_iou']:.4f}  (TinyUNet ref: 0.6049, Hybrid ref: 0.4587, Classical ref: 0.3745)")
        print(f"    Micro Prec: {agg_test['micro_precision']:.4f}  (TinyUNet ref: 0.9678, Hybrid ref: 0.7693, Classical ref: 0.5484)")
        print(f"    Micro Rec:  {agg_test['micro_recall']:.4f}  (TinyUNet ref: 0.6173, Hybrid ref: 0.5319, Classical ref: 0.5414)")
        print(f"    Micro Dice: {agg_test['micro_dice']:.4f}")

    if full_results:
        agg_full = aggregate_seg_results(full_results)
        agg_full["dataset"] = "D_Cantarell (full_dataset, 173 tiles)"
        agg_full["evaluation_type"] = "adapted (RGB-SAR to dual-pol synthesis)"
        agg_full["num_scenes"] = len(full_results)
        output["full_dataset"] = {"per_scene": full_results, "aggregate": agg_full}
        print(f"\n  Cantarell (Full Dataset, n={len(full_results)}):")
        print(f"    Micro IoU:  {agg_full['micro_iou']:.4f}  (Classical ref: 0.2924)")
        print(f"    Micro Prec: {agg_full['micro_precision']:.4f}  (Classical ref: 0.5756)")
        print(f"    Micro Rec:  {agg_full['micro_recall']:.4f}  (Classical ref: 0.3728)")

    if output:
        # Return test_split as primary aggregate for overall summary, full_dataset in per_scene
        primary = output.get("test_split", output.get("full_dataset"))
        all_scenes = (test_results if test_results else []) + (full_results if not test_results else [])
        return {
            "per_scene": all_scenes,
            "aggregate": primary["aggregate"],
            "test_aggregate": output.get("test_split", {}).get("aggregate"),
            "full_aggregate": output.get("full_dataset", {}).get("aggregate"),
        }

    return None


# ======================================================================
# Aggregation & Utilities
# ======================================================================

def aggregate_seg_results(results):
    """Compute micro and macro aggregates from per-scene results."""
    total_tp = sum(r["tp"] for r in results)
    total_fp = sum(r["fp"] for r in results)
    total_fn = sum(r["fn"] for r in results)
    total_tn = sum(r["tn"] for r in results)

    micro_dice = 2 * total_tp / max(1, 2 * total_tp + total_fp + total_fn)
    micro_iou = total_tp / max(1, total_tp + total_fp + total_fn)
    micro_precision = total_tp / max(1, total_tp + total_fp)
    micro_recall = total_tp / max(1, total_tp + total_fn)
    micro_fpr = total_fp / max(1, total_fp + total_tn)

    dices = [r["dice"] for r in results]
    ious = [r["iou"] for r in results]

    return {
        "micro_dice": micro_dice,
        "micro_iou": micro_iou,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_fpr": micro_fpr,
        "macro_mean_dice": float(np.mean(dices)),
        "macro_median_dice": float(np.median(dices)),
        "macro_mean_iou": float(np.mean(ious)),
        "macro_median_iou": float(np.median(ious)),
        "macro_std_dice": float(np.std(dices)),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_tn": total_tn,
        "total_oil_pixels": sum(r["oil_pixels_gt"] for r in results),
        "total_pixels": sum(r["total_pixels"] for r in results),
    }


def save_visual(img_path, mask, pred_prob, threshold, out_path):
    """Save a visual overlay: input | GT mask | prediction | error map."""
    try:
        img = np.array(Image.open(str(img_path)).convert("L"))
        h, w = img.shape

        # Resize mask and pred to match
        if mask.shape != img.shape:
            mask = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)) / 255.0
        if pred_prob.shape != img.shape:
            pred_prob = np.array(
                Image.fromarray((pred_prob * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
            ) / 255.0

        pred_bin = (pred_prob >= threshold).astype(np.uint8)
        mask_bin = (mask > 0).astype(np.uint8)

        # Create RGB panels
        panel_w = w
        canvas = np.zeros((h, panel_w * 4, 3), dtype=np.uint8)

        # Panel 1: Input
        canvas[:, 0:panel_w, :] = img[:, :, np.newaxis]

        # Panel 2: GT mask (green)
        gt_vis = np.zeros((h, w, 3), dtype=np.uint8)
        gt_vis[:, :, 1] = mask_bin * 255
        canvas[:, panel_w:2*panel_w, :] = gt_vis

        # Panel 3: Prediction (red)
        pred_vis = np.zeros((h, w, 3), dtype=np.uint8)
        pred_vis[:, :, 0] = pred_bin * 255
        canvas[:, 2*panel_w:3*panel_w, :] = pred_vis

        # Panel 4: Error (FP=red, FN=blue)
        err_vis = np.zeros((h, w, 3), dtype=np.uint8)
        fp_mask = (pred_bin == 1) & (mask_bin == 0)
        fn_mask = (pred_bin == 0) & (mask_bin == 1)
        tp_mask = (pred_bin == 1) & (mask_bin == 1)
        err_vis[fp_mask] = [255, 0, 0]  # Red = FP
        err_vis[fn_mask] = [0, 0, 255]  # Blue = FN
        err_vis[tp_mask] = [0, 255, 0]  # Green = TP
        canvas[:, 3*panel_w:4*panel_w, :] = err_vis

        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(canvas).save(str(out_path))
    except Exception as e:
        print(f"    [WARN] Visual save failed: {e}")


# ======================================================================
# Latency Benchmarking
# ======================================================================

def benchmark_latency(seg_model, cls_model, device):
    """Benchmark inference latency for production models."""
    print("\n" + "=" * 60)
    print("COMPUTATIONAL EFFICIENCY BENCHMARK")
    print("=" * 60)

    results = {}

    # Segmenter latency
    seg_input = torch.randn(1, 2, 512, 512).to(device)
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            seg_model(seg_input)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(20):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            seg_model(seg_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    results["segmenter_gpu_ms"] = {
        "mean": float(np.mean(times)),
        "std": float(np.std(times)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
    }
    print(f"  Segmenter GPU: {np.mean(times):.1f} +/- {np.std(times):.1f} ms")

    # CPU latency
    seg_cpu = seg_model.cpu()
    seg_cpu_input = seg_input.cpu()
    for _ in range(3):
        with torch.no_grad():
            seg_cpu(seg_cpu_input)

    cpu_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        with torch.no_grad():
            seg_cpu(seg_cpu_input)
        cpu_times.append((time.perf_counter() - t0) * 1000)

    results["segmenter_cpu_ms"] = {
        "mean": float(np.mean(cpu_times)),
        "std": float(np.std(cpu_times)),
    }
    print(f"  Segmenter CPU: {np.mean(cpu_times):.1f} +/- {np.std(cpu_times):.1f} ms")

    # Move back to GPU
    seg_model.to(device)

    # Classifier latency
    cls_input = torch.randn(1, 2, 400, 400).to(device)
    for _ in range(5):
        with torch.no_grad():
            cls_model(cls_input)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    cls_times = []
    for _ in range(20):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            cls_model(cls_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        cls_times.append((time.perf_counter() - t0) * 1000)

    results["classifier_gpu_ms"] = {
        "mean": float(np.mean(cls_times)),
        "std": float(np.std(cls_times)),
    }
    print(f"  Classifier GPU: {np.mean(cls_times):.1f} +/- {np.std(cls_times):.1f} ms")

    # ONNX Runtime latency
    try:
        import onnxruntime as ort
        onnx_seg = CONFIG.output_segmenter_onnx
        if onnx_seg.exists():
            sess = ort.InferenceSession(str(onnx_seg), providers=["CPUExecutionProvider"])
            inp_name = sess.get_inputs()[0].name
            dummy = np.random.randn(1, 2, 512, 512).astype(np.float32)

            for _ in range(3):
                sess.run(None, {inp_name: dummy})

            onnx_times = []
            for _ in range(10):
                t0 = time.perf_counter()
                sess.run(None, {inp_name: dummy})
                onnx_times.append((time.perf_counter() - t0) * 1000)

            results["segmenter_onnx_cpu_ms"] = {
                "mean": float(np.mean(onnx_times)),
                "std": float(np.std(onnx_times)),
            }
            print(f"  Segmenter ONNX CPU: {np.mean(onnx_times):.1f} +/- {np.std(onnx_times):.1f} ms")
    except Exception as e:
        print(f"  [WARN] ONNX latency test failed: {e}")

    # Model sizes
    seg_ckpt = CONFIG.checkpoint_dir / "best_segmenter_v3_exp1_lr1e4.pt"
    cls_ckpt = CONFIG.checkpoint_dir / "best_dualpol_classifier.pt"
    onnx_seg_path = CONFIG.output_segmenter_onnx
    onnx_cls_path = CONFIG.output_classifier_onnx

    results["model_sizes"] = {}
    for name, p in [("segmenter_pt", seg_ckpt), ("classifier_pt", cls_ckpt),
                    ("segmenter_onnx", onnx_seg_path), ("classifier_onnx", onnx_cls_path)]:
        if p.exists():
            results["model_sizes"][name] = p.stat().st_size / (1024 * 1024)  # MB

    # Memory usage
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            seg_model(torch.randn(1, 2, 512, 512).to(device))
        results["segmenter_peak_gpu_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return results


# ======================================================================
# Report Generation
# ======================================================================

def save_results(all_results, latency_results, seg_info, cls_info):
    """Save all benchmark results to required output files."""
    timestamp = datetime.now().isoformat()

    # 1. segmentation_results_per_scene.csv
    seg_csv = BENCHMARK_DIR / "segmentation_results_per_scene.csv"
    all_seg = []
    for ds_key, ds_result in all_results.items():
        if ds_result and "per_scene" in ds_result:
            all_seg.extend(ds_result["per_scene"])

    if all_seg:
        with open(seg_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "dataset", "scene_id", "dice", "iou", "precision", "recall",
                "f1", "fpr", "fnr", "pixel_accuracy",
                "tp", "fp", "fn", "tn", "oil_pixels_gt", "total_pixels"
            ])
            writer.writeheader()
            for r in all_seg:
                writer.writerow({k: r.get(k, "") for k in writer.fieldnames})
        print(f"  Written: {seg_csv}")

    # 2. model_comparison.csv
    mc_csv = BENCHMARK_DIR / "model_comparison.csv"
    with open(mc_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "type", "params", "checkpoint",
            "input_shape", "threshold", "onnx_size_mb",
        ])
        writer.writerow([
            "SpillSegNet", "segmenter", seg_info["params"],
            seg_info["checkpoint"], "[1,2,512,512]",
            seg_info["threshold"],
            CONFIG.output_segmenter_onnx.stat().st_size / 1e6 if CONFIG.output_segmenter_onnx.exists() else "N/A",
        ])
        writer.writerow([
            "DualPolOilSpillNet", "classifier", cls_info["params"],
            cls_info["checkpoint"], "[1,2,400,400]",
            cls_info["threshold"],
            CONFIG.output_classifier_onnx.stat().st_size / 1e6 if CONFIG.output_classifier_onnx.exists() else "N/A",
        ])
    print(f"  Written: {mc_csv}")

    # 3. latency_comparison.csv
    lat_csv = BENCHMARK_DIR / "latency_comparison.csv"
    with open(lat_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "runtime", "mean_ms", "std_ms", "min_ms", "max_ms"])
        for key, val in latency_results.items():
            if isinstance(val, dict) and "mean" in val:
                writer.writerow([key, "measured", val["mean"], val.get("std", ""),
                                val.get("min", ""), val.get("max", "")])
    print(f"  Written: {lat_csv}")

    # 4. benchmark_summary.json
    summary = {
        "timestamp": timestamp,
        "models": {
            "segmenter": seg_info,
            "classifier": cls_info,
        },
        "datasets_evaluated": list(all_results.keys()),
        "aggregates": {},
        "latency": latency_results,
    }

    for ds_key, ds_result in all_results.items():
        if ds_result and "aggregate" in ds_result:
            summary["aggregates"][ds_key] = ds_result["aggregate"]

    with open(BENCHMARK_DIR / "benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Written: benchmark_summary.json")


# ======================================================================
# Main Pipeline
# ======================================================================

def run_full_benchmark():
    """Execute the complete external validation benchmark."""
    print("=" * 70)
    print("SpillSense External Validation Benchmark Pipeline")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    device = CONFIG.get_device()
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load models
    try:
        seg_model, seg_info = load_production_segmenter(device)
    except Exception as e:
        print(f"[FATAL] Cannot load segmenter: {e}")
        return

    try:
        cls_model, cls_info = load_production_classifier(device)
    except Exception as e:
        print(f"[WARN] Cannot load classifier: {e}")
        cls_model, cls_info = None, {}

    try:
        staged_cls, staged_info = load_staged_classifier(device)
    except Exception as e:
        print(f"[WARN] Cannot load staged classifier: {e}")
        staged_cls, staged_info = None, {}

    # Run dataset evaluations
    all_results = {}

    # Dataset A: SOS
    try:
        all_results["A_SOS"] = evaluate_sos_dataset(seg_model, seg_info, device)
    except Exception as e:
        print(f"[ERROR] SOS evaluation failed: {e}")
        traceback.print_exc()
        all_results["A_SOS"] = None

    # Dataset D: Cantarell
    try:
        all_results["D_Cantarell"] = evaluate_cantarell_dataset(seg_model, seg_info, device)
    except Exception as e:
        print(f"[ERROR] Cantarell evaluation failed: {e}")
        traceback.print_exc()
        all_results["D_Cantarell"] = None

    # Latency benchmark
    try:
        latency = benchmark_latency(seg_model, cls_model if cls_model else seg_model, device)
    except Exception as e:
        print(f"[ERROR] Latency benchmark failed: {e}")
        latency = {}

    # Save all results
    save_results(all_results, latency, seg_info, cls_info)

    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    for ds_key, ds_result in all_results.items():
        if ds_result and "aggregate" in ds_result:
            agg = ds_result["aggregate"]
            print(f"\n  {ds_key}:")
            print(f"    Scenes: {agg['num_scenes']}")
            print(f"    Micro Dice:  {agg['micro_dice']:.4f}")
            print(f"    Micro IoU:   {agg['micro_iou']:.4f}")
            print(f"    Micro Prec:  {agg['micro_precision']:.4f}")
            print(f"    Micro Rec:   {agg['micro_recall']:.4f}")
            print(f"    Macro Dice:  {agg['macro_mean_dice']:.4f} +/- {agg['macro_std_dice']:.4f}")
            print(f"    Type: {agg['evaluation_type']}")
        elif ds_result is None:
            print(f"\n  {ds_key}: NOT EVALUATED (dataset unavailable or incompatible)")

    print(f"\n  Reference (internal): Dice=0.8390, IoU=0.7227")
    print("=" * 70)
    print("Benchmark complete. Results saved to ml/benchmarks/external_validation/")


if __name__ == "__main__":
    run_full_benchmark()
