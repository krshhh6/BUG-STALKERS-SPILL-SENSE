"""
SpillSense (SIH-26143) — Classifier Training Pipeline v3
=========================================================
Trains DualPolOilSpillNet using real Zenodo ground-truth masks.

Features:
- Focal Loss for hard-negative focus (gamma=2, alpha=0.25)
- Look-alike patches receive 2x loss weight
- Class-aware sampling (40% oil, 30% lookalike, 20% clean, 10% ship/wake)
- Validation threshold optimization
- Early stopping on composite metric (F1 + look-alike TNR + PR-AUC)
- ONNX export with metadata
"""

import os
import sys
import time
import json
import random
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from ml.config import CONFIG, SpillSenseConfig
from ml.dataset import SARClassificationDataset, get_or_create_cls_dataset
from ml.models import DualPolOilSpillNet, get_classifier, count_params


# ═══════════════════════════════════════════════════════
# Focal Loss
# ═══════════════════════════════════════════════════════

class FocalLossWithLogits(nn.Module):
    """
    Focal Loss (Lin et al. 2017) for binary classification.
    Down-weights easy examples, focuses on hard negatives like look-alikes.
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "none"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1.0 - p_t) ** self.gamma

        # Alpha balancing
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_t * focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ═══════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_metrics(labels: np.ndarray, probs: np.ndarray,
                    threshold: float = 0.50) -> Dict[str, float]:
    """Compute binary classification metrics."""
    preds = (probs >= threshold).astype(int)
    y = labels.astype(int)

    tp = int(np.sum((preds == 1) & (y == 1)))
    tn = int(np.sum((preds == 0) & (y == 0)))
    fp = int(np.sum((preds == 1) & (y == 0)))
    fn = int(np.sum((preds == 0) & (y == 1)))

    accuracy = (tp + tn) / max(1, len(y))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)

    # ROC-AUC via Wilcoxon-Mann-Whitney
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos > 0 and n_neg > 0:
        order = np.argsort(probs)
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        roc_auc = float((np.sum(rank[y == 1]) - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg))
    else:
        roc_auc = 0.0

    # PR-AUC via trapezoidal integration
    t_sweep = np.linspace(0.01, 0.99, 100)
    p_curve, r_curve = [], []
    for t in t_sweep:
        pr_b = (probs >= t).astype(int)
        c_tp = np.sum((pr_b == 1) & (y == 1))
        c_fp = np.sum((pr_b == 1) & (y == 0))
        c_fn = np.sum((pr_b == 0) & (y == 1))
        p_curve.append(c_tp / max(1, c_tp + c_fp))
        r_curve.append(c_tp / max(1, c_tp + c_fn))
    sorted_idx = np.argsort(r_curve)
    r_s = np.array(r_curve)[sorted_idx]
    p_s = np.array(p_curve)[sorted_idx]
    pr_auc = float(np.sum((r_s[1:] - r_s[:-1]) * (p_s[1:] + p_s[:-1]) / 2.0)) if len(r_s) > 1 else 0.0

    return {
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "f1": f1, "roc_auc": roc_auc, "pr_auc": pr_auc,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def compute_category_metrics(labels: np.ndarray, probs: np.ndarray,
                             categories: np.ndarray, threshold: float) -> Dict[str, Dict]:
    """Compute per-category metrics (TNR, FPR for negatives; recall for oil)."""
    results = {}
    preds = (probs >= threshold).astype(int)

    for cat in np.unique(categories):
        mask = categories == cat
        cat_labels = labels[mask]
        cat_preds = preds[mask]
        n = int(np.sum(mask))

        if cat == "oil":
            tp = int(np.sum((cat_preds == 1) & (cat_labels == 1)))
            fn = int(np.sum((cat_preds == 0) & (cat_labels == 1)))
            results[cat] = {
                "count": n, "recall": tp / max(1, tp + fn),
                "tp": tp, "fn": fn,
            }
        else:
            # Negative category: measure TNR (how many correctly identified as non-oil)
            tn = int(np.sum(cat_preds == 0))
            fp = int(np.sum(cat_preds == 1))
            results[cat] = {
                "count": n, "tnr": tn / max(1, n),
                "fpr": fp / max(1, n), "tn": tn, "fp": fp,
            }

    return results


def find_optimal_threshold(labels: np.ndarray, probs: np.ndarray,
                           categories: np.ndarray = None) -> Tuple[float, float]:
    """
    Find threshold maximizing composite metric:
    0.5*F1 + 0.3*look-alike_TNR + 0.2*PR-AUC
    """
    best_thresh = 0.50
    best_score = 0.0
    thresholds = np.linspace(0.05, 0.95, 46)

    for t in thresholds:
        m = compute_metrics(labels, probs, float(t))
        score = m["f1"]

        # If we have category info, weight look-alike TNR
        if categories is not None:
            cat_m = compute_category_metrics(labels, probs, categories, float(t))
            look_tnr = cat_m.get("lookalike", {}).get("tnr", 1.0)
            score = 0.5 * m["f1"] + 0.3 * look_tnr + 0.2 * m["pr_auc"]

        if score > best_score:
            best_score = score
            best_thresh = float(t)

    return best_thresh, best_score


# ═══════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp,
                    category_weights=None, categories_array=None):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                logits = model(x_batch).squeeze(-1)
                loss = criterion(logits, y_batch)
                loss = loss.mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x_batch).squeeze(-1)
            loss = criterion(logits, y_batch)
            loss = loss.mean()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x_batch.size(0)
        total_samples += x_batch.size(0)

    return total_loss / max(1, total_samples)


def evaluate_split(model, loader, criterion, device, threshold=0.50):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            logits = model(x_batch).squeeze(-1)
            loss = criterion(logits, y_batch).mean()
            probs = torch.sigmoid(logits)

            total_loss += loss.item() * x_batch.size(0)
            total_samples += x_batch.size(0)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    return (total_loss / max(1, total_samples),
            np.array(all_probs), np.array(all_labels),
            compute_metrics(np.array(all_labels), np.array(all_probs), threshold))


def export_onnx(model, path, input_size=400):
    """Export model to ONNX."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval().to("cpu")
    dummy = torch.randn(1, 2, input_size, input_size)
    torch.onnx.export(
        model, dummy, str(path), export_params=True, opset_version=14,
        do_constant_folding=True,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
    )
    size_kb = path.stat().st_size / 1024
    print(f"[ONNX] Exported to {path} ({size_kb:.1f} KB)")
    return size_kb


# ═══════════════════════════════════════════════════════
# Main Training Function
# ═══════════════════════════════════════════════════════

def train(config: SpillSenseConfig = CONFIG, force_rebuild: bool = False,
          max_scenes: Optional[int] = None,
          lr: Optional[float] = None,
          epochs: Optional[int] = None,
          patience: Optional[int] = None,
          batch_size: Optional[int] = None,
          focal_alpha: Optional[float] = None,
          focal_gamma: Optional[float] = None,
          loss_type: str = "focal",
          sample_weight_oil: Optional[float] = None,
          sample_weight_lookalike: Optional[float] = None,
          sample_weight_clean: Optional[float] = None,
          sample_weight_ship_wake: Optional[float] = None,
          out_suffix: str = "",
          seed: int = 42) -> Dict[str, Any]:
    """Main classifier training pipeline."""
    set_seed(seed)
    device = config.get_device()

    train_lr = lr if lr is not None else config.cls_learning_rate
    train_epochs = epochs if epochs is not None else config.cls_epochs
    train_patience = patience if patience is not None else config.cls_early_stopping_patience
    train_bs = batch_size if batch_size is not None else config.cls_batch_size
    alpha = focal_alpha if focal_alpha is not None else config.focal_alpha
    gamma = focal_gamma if focal_gamma is not None else config.focal_gamma

    sw_oil = sample_weight_oil if sample_weight_oil is not None else config.sample_weight_oil
    sw_look = sample_weight_lookalike if sample_weight_lookalike is not None else config.sample_weight_lookalike
    sw_clean = sample_weight_clean if sample_weight_clean is not None else config.sample_weight_clean
    sw_wake = sample_weight_ship_wake if sample_weight_ship_wake is not None else config.sample_weight_ship_wake

    print("=" * 60)
    print(f"SpillSense v3 -- Classifier Training [suffix: '{out_suffix}']")
    print(f"Config: LR={train_lr:.1e}, Epochs={train_epochs}, Patience={train_patience}, Batch={train_bs}")
    print(f"Loss: {loss_type} (alpha={alpha}, gamma={gamma})")
    print(f"Sampling weights: oil={sw_oil:.2f}, lookalike={sw_look:.2f}, clean={sw_clean:.2f}, ship_wake={sw_wake:.2f}")
    print("=" * 60)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Check dataset availability
    config.data.print_status()

    # Load dataset
    data = get_or_create_cls_dataset(config, force_rebuild=force_rebuild, max_scenes=max_scenes)
    tr_x, tr_y = data["train_patches"], data["train_labels"]
    va_x, va_y = data["val_patches"], data["val_labels"]
    tr_cats = data.get("train_categories", np.full(len(tr_y), "unknown"))
    va_cats = data.get("val_categories", np.full(len(va_y), "unknown"))

    print(f"\nTrain: {len(tr_y)} patches (Oil: {int(np.sum(tr_y==1))}, Non-oil: {int(np.sum(tr_y==0))})")
    print(f"Val:   {len(va_y)} patches (Oil: {int(np.sum(va_y==1))}, Non-oil: {int(np.sum(va_y==0))})")

    # Build weighted sampler (class-aware)
    cat_counts = {}
    for c in tr_cats:
        c_str = str(c)
        cat_counts[c_str] = cat_counts.get(c_str, 0) + 1

    sample_weights = np.zeros(len(tr_y), dtype=np.float64)
    target_fractions = {
        "oil": sw_oil,
        "lookalike": sw_look,
        "clean": sw_clean,
        "ship_wake": sw_wake,
    }
    for i in range(len(tr_y)):
        cat = str(tr_cats[i])
        count = max(1, cat_counts.get(cat, 1))
        target_frac = target_fractions.get(cat, 0.25)
        sample_weights[i] = target_frac / count

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(tr_y),
        replacement=True
    )

    # Dataloaders
    train_ds = SARClassificationDataset(tr_x, tr_y, tr_cats, augment=True)
    val_ds = SARClassificationDataset(va_x, va_y, va_cats, augment=False)

    train_loader = DataLoader(train_ds, batch_size=train_bs, sampler=sampler,
                              num_workers=config.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=train_bs, shuffle=False,
                            num_workers=config.num_workers, pin_memory=(device.type == "cuda"))

    # Model
    model = get_classifier(in_channels=config.in_channels).to(device)
    print(f"\nModel: DualPolOilSpillNet ({count_params(model):,} params)")

    # Loss criterion
    if loss_type == "bce":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = FocalLossWithLogits(gamma=gamma, alpha=alpha)

    optimizer = optim.AdamW(model.parameters(), lr=train_lr,
                            weight_decay=config.cls_weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    scaler = torch.amp.GradScaler("cuda") if (config.use_amp and device.type == "cuda") else None

    # Checkpoint paths
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = f"best_dualpol_classifier{out_suffix}.pt" if out_suffix else "best_classifier_v3.pt"
    ckpt_path = config.checkpoint_dir / ckpt_name
    history_path = config.checkpoint_dir / f"best_dualpol_classifier{out_suffix}_history.json" if out_suffix else config.checkpoint_dir / "best_classifier_v3_history.json"

    best_score = 0.0
    best_epoch = 0
    best_threshold = 0.50
    best_state = None
    patience_counter = 0
    history = []

    print(f"\n{'Ep':>3} | {'TrLoss':>7} | {'VLoss':>7} | {'VF1':>6} | {'VRec':>6} | {'VPrec':>6} | {'Thr':>5} | {'WkFPR':>5} | {'LkTNR':>5} | {'LR':>8} | {'T':>4}")
    print("-" * 95)

    for epoch in range(1, train_epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler,
                                  device, config.use_amp)
        val_loss, val_probs, val_labels, val_m = evaluate_split(
            model, val_loader, criterion, device, threshold=0.50)

        # Threshold search
        opt_thresh, opt_score = find_optimal_threshold(val_labels, val_probs, va_cats)

        # Per-category metrics at optimal threshold
        cat_m = compute_category_metrics(val_labels, val_probs, va_cats, opt_thresh)
        look_tnr = cat_m.get("lookalike", {}).get("tnr", 1.0)
        wake_fpr = cat_m.get("ship_wake", {}).get("fpr", 0.0)
        clean_fpr = cat_m.get("clean", {}).get("fpr", 0.0)

        # Also get metrics at optimal threshold
        opt_m = compute_metrics(val_labels, val_probs, opt_thresh)

        scheduler.step(opt_score)
        current_lr = float(optimizer.param_groups[0]["lr"])
        dt = time.time() - t0

        print(f"{epoch:3d} | {tr_loss:7.4f} | {val_loss:7.4f} | {opt_m['f1']:6.3f} | {opt_m['recall']:6.3f} | "
              f"{opt_m['precision']:6.3f} | {opt_thresh:5.2f} | {wake_fpr:5.2f} | {look_tnr:5.2f} | {current_lr:8.1e} | {dt:4.1f}s")

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(tr_loss),
            "val_loss": float(val_loss),
            "val_f1": float(opt_m["f1"]),
            "val_recall": float(opt_m["recall"]),
            "val_precision": float(opt_m["precision"]),
            "val_accuracy": float(opt_m["accuracy"]),
            "val_roc_auc": float(opt_m["roc_auc"]),
            "val_pr_auc": float(opt_m["pr_auc"]),
            "optimal_threshold": float(opt_thresh),
            "lookalike_tnr": float(look_tnr),
            "ship_wake_fpr": float(wake_fpr),
            "clean_fpr": float(clean_fpr),
            "lr": current_lr,
            "time_s": round(dt, 2)
        }
        history.append(epoch_record)

        if opt_score > best_score:
            best_score = opt_score
            best_epoch = epoch
            best_threshold = opt_thresh
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save({
                "epoch": epoch, "model_state": best_state,
                "best_score": best_score, "best_threshold": best_threshold,
                "val_metrics": opt_m, "category_metrics": cat_m,
                "config": {
                    "lr": train_lr, "epochs": train_epochs, "batch_size": train_bs,
                    "loss_type": loss_type, "focal_alpha": alpha, "focal_gamma": gamma,
                    "sample_weights": {"oil": sw_oil, "lookalike": sw_look, "clean": sw_clean, "ship_wake": sw_wake},
                    "seed": seed
                }
            }, ckpt_path)
            print(f"  --> Saved new best checkpoint to {ckpt_path.name} (Score: {best_score:.4f}, Thr: {best_threshold:.2f})")
        else:
            patience_counter += 1
            if patience_counter >= train_patience:
                print(f"\n[EARLY STOP] No improvement for {train_patience} epochs.")
                break

    # Save history
    with open(history_path, "w") as f:
        json.dump({
            "candidate": ckpt_name,
            "best_epoch": best_epoch,
            "best_threshold": float(best_threshold),
            "history": history
        }, f, indent=2)
    print(f"[HISTORY] Training curves saved to {history_path}")

    # Restore best
    print(f"\nRestoring best model from epoch {best_epoch}...")
    if best_state:
        model.load_state_dict(best_state)
    model.to(device)

    # Final validation eval with optimal threshold
    _, final_probs, final_labels, final_m = evaluate_split(
        model, val_loader, criterion, device, threshold=best_threshold)
    final_cats = compute_category_metrics(final_labels, final_probs, va_cats, best_threshold)

    print("\n" + "=" * 60)
    print("FINAL VALIDATION RESULTS")
    print("=" * 60)
    print(f"Threshold: {best_threshold:.2f}")
    print(f"Accuracy:  {final_m['accuracy']:.1%}")
    print(f"Precision: {final_m['precision']:.3f}")
    print(f"Recall:    {final_m['recall']:.3f}")
    print(f"F1:        {final_m['f1']:.3f}")
    print(f"PR-AUC:    {final_m['pr_auc']:.3f}")
    print(f"ROC-AUC:   {final_m['roc_auc']:.3f}")
    print(f"\nPer-Category:")
    for cat, cm in final_cats.items():
        if cat == "oil":
            print(f"  {cat:15s}: Recall={cm['recall']:.3f} (TP={cm['tp']}, FN={cm['fn']})")
        else:
            print(f"  {cat:15s}: TNR={cm['tnr']:.3f} FPR={cm['fpr']:.3f} (TN={cm['tn']}, FP={cm['fp']})")

    # Export ONNX only if not an experimental suffix
    if not out_suffix:
        onnx_kb = export_onnx(model, config.output_classifier_onnx, config.cls_patch_size)

        # Save metadata
        metadata = {
            "model_name": "DualPolOilSpillNet",
            "model_type": "classifier",
            "version": config.cache_version,
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "in_channels": config.in_channels,
            "input_shape": [1, 2, config.cls_patch_size, config.cls_patch_size],
            "channel_order": list(config.channel_names),
            "optimal_threshold": float(best_threshold),
            "normalization": {
                "vv_min_db": config.vv_min_db, "vv_max_db": config.vv_max_db,
                "vh_min_db": config.vh_min_db, "vh_max_db": config.vh_max_db,
            },
            "metrics": {
                "val_accuracy": float(final_m["accuracy"]),
                "val_precision": float(final_m["precision"]),
                "val_recall": float(final_m["recall"]),
                "val_f1": float(final_m["f1"]),
                "val_pr_auc": float(final_m["pr_auc"]),
                "val_roc_auc": float(final_m["roc_auc"]),
            },
            "category_metrics": {cat: {k: float(v) if isinstance(v, (int, float)) else v
                                        for k, v in cm.items()}
                                 for cat, cm in final_cats.items()},
            "confusion_matrix": {
                "tp": final_m["tp"], "tn": final_m["tn"],
                "fp": final_m["fp"], "fn": final_m["fn"],
            },
            "onnx_size_kb": onnx_kb,
        }
        config.model_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config.model_metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"\n[METADATA] Saved to {config.model_metadata_path}")
    else:
        print(f"[ONNX] Suffix '{out_suffix}' active — preserving production ONNX & metadata untouched.")

    return {
        "model": model, "best_epoch": best_epoch,
        "best_threshold": best_threshold, "val_metrics": final_m,
        "category_metrics": final_cats, "history": history,
        "ckpt_path": str(ckpt_path)
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train DualPolOilSpillNet")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=None, help="Max epochs")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--focal-alpha", type=float, default=None, help="Focal loss alpha")
    parser.add_argument("--focal-gamma", type=float, default=None, help="Focal loss gamma")
    parser.add_argument("--loss-type", type=str, default="focal", choices=["focal", "bce"], help="Loss function")
    parser.add_argument("--sample-weight-oil", type=float, default=None, help="Sampling weight for oil")
    parser.add_argument("--sample-weight-lookalike", type=float, default=None, help="Sampling weight for lookalike")
    parser.add_argument("--sample-weight-clean", type=float, default=None, help="Sampling weight for clean")
    parser.add_argument("--sample-weight-ship-wake", type=float, default=None, help="Sampling weight for ship_wake")
    parser.add_argument("--out-suffix", type=str, default="", help="Checkpoint suffix (e.g. _exp1_loss_alpha05)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train(
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        loss_type=args.loss_type,
        sample_weight_oil=args.sample_weight_oil,
        sample_weight_lookalike=args.sample_weight_lookalike,
        sample_weight_clean=args.sample_weight_clean,
        sample_weight_ship_wake=args.sample_weight_ship_wake,
        out_suffix=args.out_suffix,
        seed=args.seed
    )
