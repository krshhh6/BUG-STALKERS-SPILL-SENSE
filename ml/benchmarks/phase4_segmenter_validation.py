"""
SpillSense Phase 4 — Final Segmenter Validation Script
======================================================
Evaluates baseline (best_segmenter_v3.pt) and candidate (best_segmenter_v3_exp1_lr1e4.pt)
on 20 untouched, independent full-scene images (val_scenes[21:41]).

Computes:
- Per-scene: TP, FP, FN, TN, Dice, IoU, Precision, Recall, FPR, FNR, Runtime
- Aggregate Micro metrics (pooled pixels)
- Macro metrics (mean, median, std, min, max, IQR)
- Failure mode breakdown: faint slicks, small spills, tile boundaries, ship wakes
- Exports segmenter_per_scene_results.csv
"""

import sys
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.config import CONFIG, SpillSenseConfig
from ml.models import get_segmenter
from ml.preprocess import load_sentinel1_scene, load_mask, normalize_sar_dualpol
from ml.dataset import discover_scenes, partition_scenes_by_category


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()


def tiled_inference_timed(model: nn.Module, vv: np.ndarray, vh: np.ndarray,
                          patch_size: int = 512, stride: int = 384,
                          device: torch.device = None) -> Tuple[np.ndarray, float]:
    """Run tiled inference and return prob_map and time in ms."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    h, w = vv.shape
    prob_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                crop_vv = vv[y:y+patch_size, x:x+patch_size]
                crop_vh = vh[y:y+patch_size, x:x+patch_size]
                norm = normalize_sar_dualpol(crop_vv, crop_vh, CONFIG)
                inp = torch.from_numpy(norm).unsqueeze(0).float().to(device)
                logits = model(inp)
                probs = torch.sigmoid(logits).cpu().numpy()[0, 0]

                prob_map[y:y+patch_size, x:x+patch_size] += probs
                count_map[y:y+patch_size, x:x+patch_size] += 1.0

    if device.type == "cuda":
        torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1000.0

    count_map = np.maximum(count_map, 1.0)
    return prob_map / count_map, dt_ms


def compute_full_metrics(pred_prob: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.50) -> Dict[str, Any]:
    pred_bin = (pred_prob >= threshold).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)

    tp = int(np.sum((pred_bin == 1) & (gt_bin == 1)))
    fp = int(np.sum((pred_bin == 1) & (gt_bin == 0)))
    fn = int(np.sum((pred_bin == 0) & (gt_bin == 1)))
    tn = int(np.sum((pred_bin == 0) & (gt_bin == 0)))

    dice = 2.0 * tp / max(1.0, 2.0 * tp + fp + fn)
    iou = tp / max(1.0, tp + fp + fn)
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    fpr = fp / max(1.0, fp + tn)
    fnr = fn / max(1.0, fn + tp)
    pixel_acc = (tp + tn) / max(1.0, tp + tn + fp + fn)
    gt_pos = tp + fn
    total_pixels = tp + fp + fn + tn
    coverage_pct = 100.0 * gt_pos / max(1.0, total_pixels)

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "dice": dice, "iou": iou, "precision": precision, "recall": recall,
        "fpr": fpr, "fnr": fnr, "pixel_acc": pixel_acc,
        "gt_pos": gt_pos, "coverage_pct": coverage_pct,
        "total_pixels": total_pixels
    }


def analyze_tile_boundaries(prob_map: np.ndarray, gt_mask: np.ndarray, stride: int = 384, margin: int = 8) -> Dict[str, float]:
    """Check if error rates are higher along tile seams (multiples of stride)."""
    h, w = prob_map.shape
    seam_mask = np.zeros((h, w), dtype=bool)

    for y in range(stride, h, stride):
        seam_mask[max(0, y - margin):min(h, y + margin), :] = True
    for x in range(stride, w, stride):
        seam_mask[:, max(0, x - margin):min(w, x + margin)] = True

    pred_bin = (prob_map >= 0.50).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)

    # Seam error rate
    seam_errors = np.sum(pred_bin[seam_mask] != gt_bin[seam_mask])
    seam_total = np.sum(seam_mask)
    seam_err_rate = seam_errors / max(1, seam_total)

    # Interior (non-seam) error rate
    non_seam_mask = ~seam_mask
    non_seam_errors = np.sum(pred_bin[non_seam_mask] != gt_bin[non_seam_mask])
    non_seam_total = np.sum(non_seam_mask)
    non_seam_err_rate = non_seam_errors / max(1, non_seam_total)

    return {
        "seam_err_rate": float(seam_err_rate),
        "non_seam_err_rate": float(non_seam_err_rate),
        "seam_error_ratio": float(seam_err_rate / max(1e-6, non_seam_err_rate))
    }


def run_segmenter_validation():
    print("=" * 80)
    print("SPILLSENSE PHASE 4 — FULL-SCENE SEGMENTER VALIDATION AUDIT")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Discover scenes and select the 20 untouched test scenes
    scenes = discover_scenes(CONFIG)
    train_scenes, val_scenes = partition_scenes_by_category(scenes, CONFIG)
    test_scenes = val_scenes[21:41]
    print(f"\n[DATASET] Total Val Scenes: {len(val_scenes)}")
    print(f"[DATASET] Selected 20 Untouched Independent Test Scenes (val_scenes[21:41])")
    print(f"Scene IDs: {[s['scene_id'] for s in test_scenes]}\n")

    # 2. Checkpoint registry and loading
    ckpt_dir = CONFIG.checkpoint_dir
    baseline_ckpt_path = ckpt_dir / "best_segmenter_v3.pt"
    candidate_ckpt_path = ckpt_dir / "best_segmenter_v3_exp1_lr1e4.pt"

    baseline_hash = sha256_file(baseline_ckpt_path)
    candidate_hash = sha256_file(candidate_ckpt_path)

    print(f"Baseline Checkpoint:  {baseline_ckpt_path.name}")
    print(f"  SHA-256: {baseline_hash}")
    print(f"Candidate Checkpoint: {candidate_ckpt_path.name}")
    print(f"  SHA-256: {candidate_hash}\n")

    baseline_ckpt = torch.load(baseline_ckpt_path, map_location=device, weights_only=False)
    candidate_ckpt = torch.load(candidate_ckpt_path, map_location=device, weights_only=False)

    model_baseline = get_segmenter(2).to(device)
    model_baseline.load_state_dict(baseline_ckpt["model_state"])
    model_baseline.eval()

    model_candidate = get_segmenter(2).to(device)
    model_candidate.load_state_dict(candidate_ckpt["model_state"])
    model_candidate.eval()

    # Warmup models
    dummy = torch.zeros(1, 2, 512, 512, device=device)
    for _ in range(3):
        _ = model_baseline(dummy)
        _ = model_candidate(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # 3. Evaluate each scene
    results = []

    print("-" * 120)
    print(f"{'Scene ID':<12} | {'GT Pixels':>9} | {'Cov %':>6} | {'Base Dice':>9} | {'Cand Dice':>9} | {'Base IoU':>8} | {'Cand IoU':>8} | {'Base Prec':>9} | {'Cand Prec':>9} | {'Base Rec':>8} | {'Cand Rec':>8} | {'Base ms':>7} | {'Cand ms':>7}")
    print("-" * 120)

    for idx, sc in enumerate(test_scenes):
        scene_id = sc["scene_id"]
        img_path = Path(sc["image_path"])
        mask_path = Path(sc["mask_path"])

        vv, vh, valid_mask = load_sentinel1_scene(img_path)
        gt_mask = load_mask(mask_path)
        if gt_mask is None:
            gt_mask = np.zeros_like(vv, dtype=np.uint8)

        # Baseline inference
        prob_base, dt_base = tiled_inference_timed(model_baseline, vv, vh, patch_size=512, stride=384, device=device)
        m_base = compute_full_metrics(prob_base, gt_mask, threshold=0.50)
        seam_base = analyze_tile_boundaries(prob_base, gt_mask, stride=384)

        # Candidate inference
        prob_cand, dt_cand = tiled_inference_timed(model_candidate, vv, vh, patch_size=512, stride=384, device=device)
        m_cand = compute_full_metrics(prob_cand, gt_mask, threshold=0.50)
        seam_cand = analyze_tile_boundaries(prob_cand, gt_mask, stride=384)

        # Characterize scene
        is_small_spill = m_base["gt_pos"] < 50000 or m_base["coverage_pct"] < 1.0
        # Contrast check (mean vv inside spill vs outside)
        spill_vv = vv[gt_mask > 0]
        bg_vv = vv[gt_mask == 0]
        spill_mean_db = float(np.mean(spill_vv)) if len(spill_vv) > 0 else 0.0
        bg_mean_db = float(np.mean(bg_vv)) if len(bg_vv) > 0 else 0.0
        contrast_db = bg_mean_db - spill_mean_db  # positive = spill is darker than background
        is_faint_slick = contrast_db < 3.0

        row = {
            "scene_index": idx + 1,
            "scene_id": scene_id,
            "image_filename": img_path.name,
            "gt_oil_pixels": m_base["gt_pos"],
            "oil_coverage_pct": round(m_base["coverage_pct"], 3),
            "contrast_db": round(contrast_db, 2),
            "is_small_spill": is_small_spill,
            "is_faint_slick": is_faint_slick,

            # Baseline
            "baseline_tp": m_base["tp"],
            "baseline_fp": m_base["fp"],
            "baseline_fn": m_base["fn"],
            "baseline_tn": m_base["tn"],
            "baseline_dice": round(m_base["dice"], 4),
            "baseline_iou": round(m_base["iou"], 4),
            "baseline_precision": round(m_base["precision"], 4),
            "baseline_recall": round(m_base["recall"], 4),
            "baseline_fpr": float(f"{m_base['fpr']:.6f}"),
            "baseline_fnr": float(f"{m_base['fnr']:.6f}"),
            "baseline_runtime_ms": round(dt_base, 1),
            "baseline_seam_err_ratio": round(seam_base["seam_error_ratio"], 3),

            # Candidate
            "candidate_tp": m_cand["tp"],
            "candidate_fp": m_cand["fp"],
            "candidate_fn": m_cand["fn"],
            "candidate_tn": m_cand["tn"],
            "candidate_dice": round(m_cand["dice"], 4),
            "candidate_iou": round(m_cand["iou"], 4),
            "candidate_precision": round(m_cand["precision"], 4),
            "candidate_recall": round(m_cand["recall"], 4),
            "candidate_fpr": float(f"{m_cand['fpr']:.6f}"),
            "candidate_fnr": float(f"{m_cand['fnr']:.6f}"),
            "candidate_runtime_ms": round(dt_cand, 1),
            "candidate_seam_err_ratio": round(seam_cand["seam_error_ratio"], 3),

            # Deltas
            "dice_delta": round(m_cand["dice"] - m_base["dice"], 4),
            "iou_delta": round(m_cand["iou"] - m_base["iou"], 4),
            "precision_delta": round(m_cand["precision"] - m_base["precision"], 4),
            "recall_delta": round(m_cand["recall"] - m_base["recall"], 4),
            "fp_pixel_reduction": int(m_base["fp"] - m_cand["fp"]),
            "fp_reduction_pct": round(100.0 * (m_base["fp"] - m_cand["fp"]) / max(1, m_base["fp"]), 2),
        }
        results.append(row)

        print(f"{scene_id:<12} | {m_base['gt_pos']:9,d} | {m_base['coverage_pct']:5.2f}% | {m_base['dice']:9.4f} | {m_cand['dice']:9.4f} | {m_base['iou']:8.4f} | {m_cand['iou']:8.4f} | {m_base['precision']:9.4f} | {m_cand['precision']:9.4f} | {m_base['recall']:8.4f} | {m_cand['recall']:8.4f} | {dt_base:7.1f} | {dt_cand:7.1f}")

    print("-" * 120)

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save per-scene CSV
    csv_path = ROOT / "segmenter_per_scene_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[OUTPUT] Saved per-scene results to: {csv_path}")

    # Micro Aggregate Calculations (Pooled pixels across all 20 scenes)
    tot_base_tp = df["baseline_tp"].sum()
    tot_base_fp = df["baseline_fp"].sum()
    tot_base_fn = df["baseline_fn"].sum()
    tot_base_tn = df["baseline_tn"].sum()

    tot_cand_tp = df["candidate_tp"].sum()
    tot_cand_fp = df["candidate_fp"].sum()
    tot_cand_fn = df["candidate_fn"].sum()
    tot_cand_tn = df["candidate_tn"].sum()

    micro_base_dice = 2.0 * tot_base_tp / (2.0 * tot_base_tp + tot_base_fp + tot_base_fn)
    micro_base_iou = tot_base_tp / (tot_base_tp + tot_base_fp + tot_base_fn)
    micro_base_prec = tot_base_tp / (tot_base_tp + tot_base_fp)
    micro_base_rec = tot_base_tp / (tot_base_tp + tot_base_fn)
    micro_base_fpr = tot_base_fp / (tot_base_fp + tot_base_tn)
    micro_base_fnr = tot_base_fn / (tot_base_fn + tot_base_tp)

    micro_cand_dice = 2.0 * tot_cand_tp / (2.0 * tot_cand_tp + tot_cand_fp + tot_cand_fn)
    micro_cand_iou = tot_cand_tp / (tot_cand_tp + tot_cand_fp + tot_cand_fn)
    micro_cand_prec = tot_cand_tp / (tot_cand_tp + tot_cand_fp)
    micro_cand_rec = tot_cand_tp / (tot_cand_tp + tot_cand_fn)
    micro_cand_fpr = tot_cand_fp / (tot_cand_fp + tot_cand_tn)
    micro_cand_fnr = tot_cand_fn / (tot_cand_fn + tot_cand_tp)

    # Macro Statistics
    macro_stats = {
        "baseline": {
            "dice": {"mean": df["baseline_dice"].mean(), "median": df["baseline_dice"].median(), "std": df["baseline_dice"].std(), "min": df["baseline_dice"].min(), "max": df["baseline_dice"].max(), "iqr": df["baseline_dice"].quantile(0.75) - df["baseline_dice"].quantile(0.25)},
            "iou": {"mean": df["baseline_iou"].mean(), "median": df["baseline_iou"].median(), "std": df["baseline_iou"].std(), "min": df["baseline_iou"].min(), "max": df["baseline_iou"].max()},
            "precision": {"mean": df["baseline_precision"].mean(), "median": df["baseline_precision"].median(), "std": df["baseline_precision"].std(), "min": df["baseline_precision"].min(), "max": df["baseline_precision"].max()},
            "recall": {"mean": df["baseline_recall"].mean(), "median": df["baseline_recall"].median(), "std": df["baseline_recall"].std(), "min": df["baseline_recall"].min(), "max": df["baseline_recall"].max()},
            "fpr": {"mean": df["baseline_fpr"].mean(), "median": df["baseline_fpr"].median(), "std": df["baseline_fpr"].std()},
            "fnr": {"mean": df["baseline_fnr"].mean(), "median": df["baseline_fnr"].median(), "std": df["baseline_fnr"].std()},
            "runtime_ms": {"mean": df["baseline_runtime_ms"].mean(), "median": df["baseline_runtime_ms"].median(), "std": df["baseline_runtime_ms"].std(), "p95": df["baseline_runtime_ms"].quantile(0.95)},
        },
        "candidate": {
            "dice": {"mean": df["candidate_dice"].mean(), "median": df["candidate_dice"].median(), "std": df["candidate_dice"].std(), "min": df["candidate_dice"].min(), "max": df["candidate_dice"].max(), "iqr": df["candidate_dice"].quantile(0.75) - df["candidate_dice"].quantile(0.25)},
            "iou": {"mean": df["candidate_iou"].mean(), "median": df["candidate_iou"].median(), "std": df["candidate_iou"].std(), "min": df["candidate_iou"].min(), "max": df["candidate_iou"].max()},
            "precision": {"mean": df["candidate_precision"].mean(), "median": df["candidate_precision"].median(), "std": df["candidate_precision"].std(), "min": df["candidate_precision"].min(), "max": df["candidate_precision"].max()},
            "recall": {"mean": df["candidate_recall"].mean(), "median": df["candidate_recall"].median(), "std": df["candidate_recall"].std(), "min": df["candidate_recall"].min(), "max": df["candidate_recall"].max()},
            "fpr": {"mean": df["candidate_fpr"].mean(), "median": df["candidate_fpr"].median(), "std": df["candidate_fpr"].std()},
            "fnr": {"mean": df["candidate_fnr"].mean(), "median": df["candidate_fnr"].median(), "std": df["candidate_fnr"].std()},
            "runtime_ms": {"mean": df["candidate_runtime_ms"].mean(), "median": df["candidate_runtime_ms"].median(), "std": df["candidate_runtime_ms"].std(), "p95": df["candidate_runtime_ms"].quantile(0.95)},
        },
        "micro_aggregate": {
            "baseline": {
                "tp": int(tot_base_tp), "fp": int(tot_base_fp), "fn": int(tot_base_fn), "tn": int(tot_base_tn),
                "dice": float(micro_base_dice), "iou": float(micro_base_iou),
                "precision": float(micro_base_prec), "recall": float(micro_base_rec),
                "fpr": float(micro_base_fpr), "fnr": float(micro_base_fnr)
            },
            "candidate": {
                "tp": int(tot_cand_tp), "fp": int(tot_cand_fp), "fn": int(tot_cand_fn), "tn": int(tot_cand_tn),
                "dice": float(micro_cand_dice), "iou": float(micro_cand_iou),
                "precision": float(micro_cand_prec), "recall": float(micro_cand_rec),
                "fpr": float(micro_cand_fpr), "fnr": float(micro_cand_fnr)
            }
        }
    }

    # Summary Display
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON (20 UNTOUCHED FULL SCENES — 83,886,080 PIXELS)")
    print("=" * 80)
    print(f"{'Metric':<30} | {'Baseline (Epoch 2)':<20} | {'Candidate (Epoch 29)':<20} | {'Delta':<12}")
    print("-" * 88)
    print(f"{'Micro (Aggregate) Dice':<30} | {micro_base_dice:20.4f} | {micro_cand_dice:20.4f} | {micro_cand_dice - micro_base_dice:+12.4f}")
    print(f"{'Micro (Aggregate) IoU':<30} | {micro_base_iou:20.4f} | {micro_cand_iou:20.4f} | {micro_cand_iou - micro_base_iou:+12.4f}")
    print(f"{'Micro (Aggregate) Precision':<30} | {micro_base_prec:20.4f} | {micro_cand_prec:20.4f} | {micro_cand_prec - micro_base_prec:+12.4f}")
    print(f"{'Micro (Aggregate) Recall':<30} | {micro_base_rec:20.4f} | {micro_cand_rec:20.4f} | {micro_cand_rec - micro_base_rec:+12.4f}")
    print(f"{'Total False Positive Pixels':<30} | {tot_base_fp:20,d} | {tot_cand_fp:20,d} | {tot_cand_fp - tot_base_fp:+12,d} ({100*(tot_cand_fp-tot_base_fp)/tot_base_fp:+.1f}%)")
    print(f"{'Total True Positive Pixels':<30} | {tot_base_tp:20,d} | {tot_cand_tp:20,d} | {tot_cand_tp - tot_base_tp:+12,d}")
    print(f"{'Total False Negative Pixels':<30} | {tot_base_fn:20,d} | {tot_cand_fn:20,d} | {tot_cand_fn - tot_base_fn:+12,d}")
    print("-" * 88)
    print(f"{'Macro Mean Dice':<30} | {macro_stats['baseline']['dice']['mean']:20.4f} | {macro_stats['candidate']['dice']['mean']:20.4f} | {macro_stats['candidate']['dice']['mean'] - macro_stats['baseline']['dice']['mean']:+12.4f}")
    print(f"{'Macro Median Dice':<30} | {macro_stats['baseline']['dice']['median']:20.4f} | {macro_stats['candidate']['dice']['median']:20.4f} | {macro_stats['candidate']['dice']['median'] - macro_stats['baseline']['dice']['median']:+12.4f}")
    print(f"{'Dice Std Dev (Variability)':<30} | {macro_stats['baseline']['dice']['std']:20.4f} | {macro_stats['candidate']['dice']['std']:20.4f} | {macro_stats['candidate']['dice']['std'] - macro_stats['baseline']['dice']['std']:+12.4f}")
    print(f"{'Dice IQR':<30} | {macro_stats['baseline']['dice']['iqr']:20.4f} | {macro_stats['candidate']['dice']['iqr']:20.4f} | {macro_stats['candidate']['dice']['iqr'] - macro_stats['baseline']['dice']['iqr']:+12.4f}")
    print(f"{'Macro Mean IoU':<30} | {macro_stats['baseline']['iou']['mean']:20.4f} | {macro_stats['candidate']['iou']['mean']:20.4f} | {macro_stats['candidate']['iou']['mean'] - macro_stats['baseline']['iou']['mean']:+12.4f}")
    print(f"{'Macro Mean Precision':<30} | {macro_stats['baseline']['precision']['mean']:20.4f} | {macro_stats['candidate']['precision']['mean']:20.4f} | {macro_stats['candidate']['precision']['mean'] - macro_stats['baseline']['precision']['mean']:+12.4f}")
    print(f"{'Macro Mean Recall':<30} | {macro_stats['baseline']['recall']['mean']:20.4f} | {macro_stats['candidate']['recall']['mean']:20.4f} | {macro_stats['candidate']['recall']['mean'] - macro_stats['baseline']['recall']['mean']:+12.4f}")
    print(f"{'Macro Mean FPR':<30} | {macro_stats['baseline']['fpr']['mean']:20.6f} | {macro_stats['candidate']['fpr']['mean']:20.6f} | {macro_stats['candidate']['fpr']['mean'] - macro_stats['baseline']['fpr']['mean']:+12.6f}")
    print(f"{'Macro Mean FNR':<30} | {macro_stats['baseline']['fnr']['mean']:20.6f} | {macro_stats['candidate']['fnr']['mean']:20.6f} | {macro_stats['candidate']['fnr']['mean'] - macro_stats['baseline']['fnr']['mean']:+12.6f}")
    print(f"{'Mean Runtime / Scene (ms)':<30} | {macro_stats['baseline']['runtime_ms']['mean']:20.1f} | {macro_stats['candidate']['runtime_ms']['mean']:20.1f} | {macro_stats['candidate']['runtime_ms']['mean'] - macro_stats['baseline']['runtime_ms']['mean']:+12.1f}")
    print("=" * 80)

    # Failure Mode Analysis
    print("\nFAILURE MODE BREAKDOWN:")
    # Small spills
    small_df = df[df["is_small_spill"]]
    print(f"\n1. Small Spills (N={len(small_df)} scenes, <50k pixels / <1% coverage):")
    print(f"   Baseline Mean Dice:  {small_df['baseline_dice'].mean():.4f} | Precision: {small_df['baseline_precision'].mean():.4f}")
    print(f"   Candidate Mean Dice: {small_df['candidate_dice'].mean():.4f} | Precision: {small_df['candidate_precision'].mean():.4f}")
    print(f"   Gain: Dice {small_df['candidate_dice'].mean() - small_df['baseline_dice'].mean():+.4f}, Precision {small_df['candidate_precision'].mean() - small_df['baseline_precision'].mean():+.4f}")

    # Faint slicks
    faint_df = df[df["is_faint_slick"]]
    print(f"\n2. Faint Slicks (N={len(faint_df)} scenes, contrast < 3.0 dB):")
    if len(faint_df) > 0:
        print(f"   Baseline Mean Dice:  {faint_df['baseline_dice'].mean():.4f} | Recall: {faint_df['baseline_recall'].mean():.4f}")
        print(f"   Candidate Mean Dice: {faint_df['candidate_dice'].mean():.4f} | Recall: {faint_df['candidate_recall'].mean():.4f}")
    else:
        print("   No scenes with contrast < 3.0 dB in this partition.")

    # Tile Boundary Seam Analysis
    print(f"\n3. Tile Boundary Seams (384-pixel stride seams vs interior):")
    print(f"   Baseline Seam Error Ratio:  {df['baseline_seam_err_ratio'].mean():.3f}x")
    print(f"   Candidate Seam Error Ratio: {df['candidate_seam_err_ratio'].mean():.3f}x")
    print(f"   (A ratio ~1.0 indicates smooth transition with no boundary discontinuities).")

    # Export json summary for documentation
    summary_path = ROOT / "segmenter_validation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(macro_stats, f, indent=2)
    print(f"\n[OUTPUT] Saved summary metrics to: {summary_path}")


if __name__ == "__main__":
    run_segmenter_validation()
