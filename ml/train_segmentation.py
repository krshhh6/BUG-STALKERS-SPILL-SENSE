"""
SpillSense (SIH-26143) — Segmentation Training Pipeline
========================================================
Trains SpillSegNet U-Net using real Zenodo ground-truth masks.

Features:
- Dice + BCE combined loss
- Oil-containing crops oversampled
- 512x512 random crops from 2048x2048 scenes
- Tiled inference for full-scene evaluation
- ONNX export
"""

import time
import sys
import json
import random
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from ml.config import CONFIG, SpillSenseConfig
from ml.dataset import (
    CachedSARSegmentationDataset, build_cached_seg_split,
    discover_scenes, partition_scenes_by_category
)
from ml.models import SpillSegNet, get_segmenter, count_params
from ml.preprocess import load_sentinel1_scene, load_mask, normalize_sar_dualpol


# ═══════════════════════════════════════════════════════
# Dice + BCE Combined Loss
# ═══════════════════════════════════════════════════════

class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss for segmentation."""
    def __init__(self, dice_weight: float = 0.6, bce_weight: float = 0.4, smooth: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE component
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")

        # Dice component
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        dice_loss = 1.0 - dice

        return self.dice_weight * dice_loss + self.bce_weight * bce


# ═══════════════════════════════════════════════════════
# Segmentation Metrics
# ═══════════════════════════════════════════════════════

def compute_seg_metrics(preds: np.ndarray, targets: np.ndarray,
                        threshold: float = 0.5) -> Dict[str, float]:
    """Compute segmentation metrics: Dice, IoU, Precision, Recall."""
    pred_binary = (preds >= threshold).astype(int)
    target_binary = (targets > 0).astype(int)

    tp = np.sum((pred_binary == 1) & (target_binary == 1))
    fp = np.sum((pred_binary == 1) & (target_binary == 0))
    fn = np.sum((pred_binary == 0) & (target_binary == 1))
    tn = np.sum((pred_binary == 0) & (target_binary == 0))

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    dice = 2 * tp / max(1, 2 * tp + fp + fn)
    iou = tp / max(1, tp + fp + fn)
    pixel_acc = (tp + tn) / max(1, tp + tn + fp + fn)

    return {
        "dice": dice, "iou": iou,
        "precision": precision, "recall": recall,
        "pixel_accuracy": pixel_acc,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


# ═══════════════════════════════════════════════════════
# Tiled Full-Scene Inference
# ═══════════════════════════════════════════════════════

def tiled_inference(model: nn.Module, vv: np.ndarray, vh: np.ndarray,
                    config: SpillSenseConfig = CONFIG, device: torch.device = None) -> np.ndarray:
    """
    Run segmentation on full 2048x2048 scene using 512x512 tiles.
    Returns probability map of same size as input.
    """
    if device is None:
        device = config.get_device()

    model.eval()
    h, w = vv.shape
    ps = config.seg_patch_size
    stride = config.seg_stride

    prob_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    with torch.no_grad():
        for y in range(0, h - ps + 1, stride):
            for x in range(0, w - ps + 1, stride):
                crop_vv = vv[y:y+ps, x:x+ps]
                crop_vh = vh[y:y+ps, x:x+ps]
                norm = normalize_sar_dualpol(crop_vv, crop_vh, config)

                inp = torch.from_numpy(norm).unsqueeze(0).float().to(device)
                logits = model(inp)
                probs = torch.sigmoid(logits).cpu().numpy()[0, 0]

                prob_map[y:y+ps, x:x+ps] += probs
                count_map[y:y+ps, x:x+ps] += 1.0

    # Average overlapping predictions
    count_map = np.maximum(count_map, 1.0)
    return prob_map / count_map


# ═══════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════

def train_segmentation(config: SpillSenseConfig = CONFIG,
                       max_scenes: Optional[int] = 120,
                       lr: Optional[float] = None,
                       patience: Optional[int] = None,
                       epochs: Optional[int] = None,
                       out_suffix: str = "",
                       seed: int = 42) -> Dict[str, Any]:
    """Main segmentation training pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = config.get_device()

    seg_lr = lr if lr is not None else config.seg_learning_rate
    seg_patience = patience if patience is not None else config.seg_early_stopping_patience
    seg_epochs = epochs if epochs is not None else config.seg_epochs

    print("=" * 60)
    print(f"SpillSense v3 -- Segmentation Training (Zenodo GT) [suffix: '{out_suffix}']")
    print(f"Config: LR={seg_lr:.1e}, Patience={seg_patience}, Epochs={seg_epochs}, Seed={seed}")
    print("=" * 60)
    print(f"Device: {device}\n")

    # Discover scenes and partition
    scenes = discover_scenes(config)
    train_scenes, val_scenes = partition_scenes_by_category(scenes, config)

    if max_scenes:
        train_scenes = train_scenes[:int(max_scenes * 0.82)]
        val_scenes = val_scenes[:int(max_scenes * 0.18)]
    else:
        train_scenes = train_scenes[:300]
        val_scenes = val_scenes[:60]

    tr_crops, tr_masks = build_cached_seg_split(train_scenes, "Train", config, oil_crops_per_scene=2, neg_crops_per_scene=1)
    va_crops, va_masks = build_cached_seg_split(val_scenes, "Val", config, oil_crops_per_scene=2, neg_crops_per_scene=1)

    train_ds = CachedSARSegmentationDataset(tr_crops, tr_masks, augment=True)
    val_ds = CachedSARSegmentationDataset(va_crops, va_masks, augment=False)

    print(f"Train crops: {len(train_ds)}")
    print(f"Val crops:   {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=config.seg_batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=config.seg_batch_size, shuffle=False,
                            num_workers=0, pin_memory=(device.type == "cuda"))

    # Model
    model = get_segmenter(in_channels=config.in_channels).to(device)
    print(f"\nModel: SpillSegNet ({count_params(model):,} params)")

    # Loss
    criterion = DiceBCELoss(dice_weight=config.seg_dice_weight, bce_weight=config.seg_bce_weight)

    optimizer = optim.AdamW(model.parameters(), lr=seg_lr,
                            weight_decay=config.seg_weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    scaler = torch.amp.GradScaler("cuda") if (config.use_amp and device.type == "cuda") else None

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = f"best_segmenter_v3{out_suffix}.pt"
    ckpt_path = config.checkpoint_dir / ckpt_name
    history_path = config.checkpoint_dir / f"best_segmenter_v3{out_suffix}_history.json"

    best_dice = 0.0
    best_epoch = 0
    best_state = None
    patience_counter = 0
    history = []

    print(f"\n{'Ep':>3} | {'TrLoss':>7} | {'VLoss':>7} | {'VDice':>6} | {'VIoU':>6} | {'VPrec':>6} | {'VRec':>6} | {'LR':>8} | {'T':>4}")
    print("-" * 80)

    for epoch in range(1, config.seg_epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0.0
        train_samples = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if config.use_amp and device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    logits = model(x_batch)
                    loss = criterion(logits, y_batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * x_batch.size(0)
            train_samples += x_batch.size(0)

        avg_tr_loss = train_loss / max(1, train_samples)

        # Validate
        model.eval()
        val_loss = 0.0
        val_samples = 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                probs = torch.sigmoid(logits)

                val_loss += loss.item() * x_batch.size(0)
                val_samples += x_batch.size(0)
                all_preds.append(probs.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())

        avg_val_loss = val_loss / max(1, val_samples)
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        seg_m = compute_seg_metrics(all_preds, all_targets)

        scheduler.step(seg_m["dice"])
        current_lr = float(optimizer.param_groups[0]["lr"])
        dt = time.time() - t0

        print(f"{epoch:3d} | {avg_tr_loss:7.4f} | {avg_val_loss:7.4f} | {seg_m['dice']:6.3f} | "
              f"{seg_m['iou']:6.3f} | {seg_m['precision']:6.3f} | {seg_m['recall']:6.3f} | {current_lr:8.1e} | {dt:4.1f}s")

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(avg_tr_loss),
            "val_loss": float(avg_val_loss),
            "dice": float(seg_m["dice"]),
            "iou": float(seg_m["iou"]),
            "precision": float(seg_m["precision"]),
            "recall": float(seg_m["recall"]),
            "pixel_accuracy": float(seg_m["pixel_accuracy"]),
            "lr": current_lr,
            "time_s": round(dt, 2)
        }
        history.append(epoch_record)

        if seg_m["dice"] > best_dice:
            best_dice = seg_m["dice"]
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save({
                "epoch": epoch, "model_state": best_state,
                "best_dice": best_dice, "seg_metrics": seg_m,
                "config": {
                    "lr": seg_lr, "patience": seg_patience,
                    "max_scenes": max_scenes, "seed": seed
                }
            }, ckpt_path)
            print(f"  --> Saved new best checkpoint to {ckpt_path.name} (Dice: {best_dice:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= seg_patience:
                print(f"\n[EARLY STOP] No improvement for {seg_patience} epochs.")
                break

    # Save training history curve
    with open(history_path, "w") as f:
        json.dump({
            "candidate": ckpt_name,
            "best_epoch": best_epoch,
            "best_dice": float(best_dice),
            "history": history
        }, f, indent=2)
    print(f"[HISTORY] Training curves saved to {history_path}")

    # Restore best
    if best_state:
        model.load_state_dict(best_state)
    model.to(device)

    print(f"\nBest epoch: {best_epoch}, Best Dice: {best_dice:.4f}")

    # Export ONNX only if not an experimental suffix (preserves production onnx)
    if not out_suffix:
        model.eval().to("cpu")
        config.output_segmenter_onnx.parent.mkdir(parents=True, exist_ok=True)
        dummy = torch.randn(1, 2, config.seg_patch_size, config.seg_patch_size)
        torch.onnx.export(
            model, dummy, str(config.output_segmenter_onnx),
            export_params=True, opset_version=14, do_constant_folding=True,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
        )
        seg_size_kb = config.output_segmenter_onnx.stat().st_size / 1024
        print(f"[ONNX] Segmenter exported ({seg_size_kb:.1f} KB)")
    else:
        print(f"[ONNX] Suffix '{out_suffix}' active — preserving production ONNX untouched.")

    return {
        "model": model, "best_epoch": best_epoch,
        "best_dice": best_dice, "history": history, "ckpt_path": str(ckpt_path)
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train SpillSegNet")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: from config 5e-4)")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience (default: from config 6)")
    parser.add_argument("--epochs", type=int, default=None, help="Max epochs (default: from config 30)")
    parser.add_argument("--max-scenes", type=int, default=120, help="Scene count (default 120, matching baseline split)")
    parser.add_argument("--out-suffix", type=str, default="", help="Checkpoint name suffix (e.g. _exp1_lr1e4)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_segmentation(
        lr=args.lr,
        patience=args.patience,
        epochs=args.epochs,
        max_scenes=args.max_scenes,
        out_suffix=args.out_suffix,
        seed=args.seed
    )
