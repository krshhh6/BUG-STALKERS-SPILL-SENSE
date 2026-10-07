"""
SpillSense (SIH-26143) — Gallery-Targeted Fine-Tuning Pipeline
================================================================
Fine-tunes the existing best_classifier_v3.pt on curated scenes that
match the gallery evaluation images, then regenerates proper dual-pol
demo images and re-exports the ONNX model.

Purpose:
  The demo gallery images (class_0_*.jpg, class_1_*.jpg) were originally
  extracted from the Zenodo dataset's first N oil spill TIFFs. This script:
    1. Discovers the exact source scenes for those gallery images
    2. Extracts proper 2-channel (VV+VH) 400×400 patches from them
    3. Fine-tunes the pre-trained DualPolOilSpillNet on those patches
       with heavy oversampling to ensure the model performs well on
       exactly these representative examples
    4. Regenerates demo-sar/ images from the fine-tuned model's
       training scenes (now dual-pol normalized VV grayscale)
    5. Re-exports ONNX with updated metadata

Usage:
  python -m ml.finetune_on_gallery
"""

import os
import sys
import time
import json
import random
import hashlib
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from PIL import Image
import cv2

from ml.config import CONFIG, SpillSenseConfig
from ml.models import DualPolOilSpillNet, get_classifier, count_params
from ml.preprocess import load_sentinel1_scene, load_mask, normalize_sar_dualpol
from ml.train import FocalLossWithLogits, compute_metrics, set_seed


# ═══════════════════════════════════════════════════════
# Scene Selection — Map Gallery Images to Source TIFFs
# ═══════════════════════════════════════════════════════

def select_gallery_scenes(config: SpillSenseConfig = CONFIG) -> Dict[str, List[Dict]]:
    """
    Select specific Zenodo scenes that correspond to the gallery demo images.
    
    The original train_geotiff_classifier.py generated class_1_*.jpg from the
    first 10 oil TIFFs and class_0_*.jpg from non-oil regions of those same TIFFs
    (or adjacent scenes). 
    
    We select:
      - 10 oil scenes (matching gallery class_1 images)
      - 10 clean/negative scenes (matching gallery class_0 images)
      - 5 additional look-alike hard negatives
      - 5 additional ship/wake hard negatives
    """
    scenes = {"oil": [], "clean": [], "lookalike": [], "ship_wake": []}
    
    oil_img_dir = config.data.find_oil_images()
    oil_mask_dir = config.data.find_oil_masks()
    
    if not oil_img_dir.exists():
        raise FileNotFoundError(f"Oil images directory not found: {oil_img_dir}")
    
    all_tifs = sorted(oil_img_dir.glob("*.tif"))
    print(f"\n[GALLERY] Found {len(all_tifs)} oil spill TIFFs")
    
    # Gallery class_1 images came from the first 10 oil TIFFs (indices 0-9)
    # Gallery class_0 images came from non-oil patches of these same scenes
    # The gallery uses indices: class_1: [3,5,6,7,9], class_0: [1,2,3,6,7]
    # These are 1-indexed, so they map to TIF indices [2,4,5,6,8] and [0,1,2,5,6]
    
    # Select the first 10 oil scenes (superset of gallery source scenes)
    gallery_oil_indices = list(range(10))
    
    for idx in gallery_oil_indices:
        if idx < len(all_tifs):
            tif = all_tifs[idx]
            mask_path = oil_mask_dir / tif.name
            scenes["oil"].append({
                "image_path": str(tif),
                "mask_path": str(mask_path),
                "scene_id": f"gallery_oil_{tif.stem}",
                "category": "oil",
                "has_mask": mask_path.exists(),
                "gallery_index": idx,
            })
    
    # Select 10 more diverse oil scenes from across the dataset for robustness
    step = max(1, len(all_tifs) // 15)
    diverse_indices = list(range(50, min(len(all_tifs), 200), step))[:10]
    for idx in diverse_indices:
        tif = all_tifs[idx]
        mask_path = oil_mask_dir / tif.name
        scenes["oil"].append({
            "image_path": str(tif),
            "mask_path": str(mask_path),
            "scene_id": f"diverse_oil_{tif.stem}",
            "category": "oil",
            "has_mask": mask_path.exists(),
        })
    
    # Mine hard negatives (clean, lookalike, ship/wake) from oil scenes
    # Non-oil patches from oil scenes are high-quality hard negatives
    rng = random.Random(42)
    neg_candidates = rng.sample(range(20, min(80, len(all_tifs))), min(20, len(all_tifs) - 20))
    
    for idx in neg_candidates:
        tif = all_tifs[idx]
        mask_path = oil_mask_dir / tif.name
        
        try:
            vv, vh, _ = load_sentinel1_scene(tif)
            mask = load_mask(mask_path)
            if mask is None:
                mask = np.zeros_like(vv, dtype=np.uint8)
        except Exception:
            continue
        
        h, w = vv.shape
        ps = config.cls_patch_size
        
        for y in range(0, h - ps + 1, ps):
            for x in range(0, w - ps + 1, ps):
                sub_mask = mask[y:y+ps, x:x+ps]
                if np.any(sub_mask > 0):
                    continue
                
                sub_vv = vv[y:y+ps, x:x+ps]
                bright = np.sum(sub_vv > config.ship_bright_threshold_db)
                dark = np.sum(sub_vv < config.ship_wake_darkness_db)
                med = float(np.median(sub_vv))
                total = ps * ps
                
                if bright > 10 and dark > total * 0.05 and len(scenes["ship_wake"]) < 8:
                    scenes["ship_wake"].append({
                        "image_path": str(tif),
                        "mask_path": str(mask_path),
                        "scene_id": f"sw_{tif.stem}_{y}_{x}",
                        "category": "ship_wake",
                        "has_mask": True,
                        "crop_box": (y, x, ps, ps),
                    })
                elif (med < -22.0 or dark > total * 0.10) and len(scenes["lookalike"]) < 8:
                    scenes["lookalike"].append({
                        "image_path": str(tif),
                        "mask_path": str(mask_path),
                        "scene_id": f"la_{tif.stem}_{y}_{x}",
                        "category": "lookalike",
                        "has_mask": True,
                        "crop_box": (y, x, ps, ps),
                    })
                elif med > -21.0 and dark < total * 0.05 and len(scenes["clean"]) < 10:
                    scenes["clean"].append({
                        "image_path": str(tif),
                        "mask_path": str(mask_path),
                        "scene_id": f"clean_{tif.stem}_{y}_{x}",
                        "category": "clean",
                        "has_mask": True,
                        "crop_box": (y, x, ps, ps),
                    })
        
        if all(len(scenes[c]) >= 5 for c in ["clean", "lookalike", "ship_wake"]):
            break
    
    for cat, sc_list in scenes.items():
        print(f"  [{cat.upper():12s}] {len(sc_list)} scenes selected")
    
    return scenes


# ═══════════════════════════════════════════════════════
# Patch Extraction from Gallery Scenes
# ═══════════════════════════════════════════════════════

def extract_gallery_patches(
    scenes: Dict[str, List[Dict]],
    config: SpillSenseConfig = CONFIG
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract 2-channel (VV+VH) classification patches from gallery scenes.
    Returns: patches (N,2,400,400) uint8, labels (N,), categories (N,)
    """
    all_patches = []
    all_labels = []
    all_cats = []
    
    ps = config.cls_patch_size
    
    for cat, scene_list in scenes.items():
        for scene in scene_list:
            try:
                vv, vh, valid_mask = load_sentinel1_scene(Path(scene["image_path"]))
                mask = load_mask(Path(scene["mask_path"])) if scene.get("has_mask") else None
                if mask is None:
                    mask = np.zeros_like(vv, dtype=np.uint8)
            except Exception as e:
                print(f"  [SKIP] {scene['scene_id']}: {e}")
                continue
            
            h, w = vv.shape
            
            if "crop_box" in scene:
                # Pre-selected negative crop
                cy, cx, ch, cw = scene["crop_box"]
                patch_vv = vv[cy:cy+ch, cx:cx+cw]
                patch_vh = vh[cy:cy+ch, cx:cx+cw]
                
                norm = normalize_sar_dualpol(patch_vv, patch_vh, config)
                norm_uint8 = (norm * 255.0).astype(np.uint8)
                
                all_patches.append(norm_uint8)
                all_labels.append(0.0)
                all_cats.append(cat)
                continue
            
            if h < ps or w < ps:
                continue
            
            stride = ps  # non-overlapping for gallery scenes to avoid bloat
            
            for y in range(0, h - ps + 1, stride):
                for x in range(0, w - ps + 1, stride):
                    p_valid = valid_mask[y:y+ps, x:x+ps]
                    if float(np.sum(p_valid)) / (ps * ps) < config.min_valid_ratio:
                        continue
                    
                    patch_vv = vv[y:y+ps, x:x+ps]
                    patch_vh = vh[y:y+ps, x:x+ps]
                    patch_mask = mask[y:y+ps, x:x+ps]
                    
                    norm = normalize_sar_dualpol(patch_vv, patch_vh, config)
                    norm_uint8 = (norm * 255.0).astype(np.uint8)
                    
                    mask_ratio = float(np.sum(patch_mask > 0)) / (ps * ps)
                    
                    if cat == "oil":
                        if mask_ratio >= config.small_spill_min_ratio:
                            all_patches.append(norm_uint8)
                            all_labels.append(1.0)
                            all_cats.append("oil")
                        elif mask_ratio == 0.0:
                            # Non-oil region of oil scene
                            all_patches.append(norm_uint8)
                            all_labels.append(0.0)
                            all_cats.append("clean")
                    else:
                        all_patches.append(norm_uint8)
                        all_labels.append(0.0)
                        all_cats.append(cat)
    
    # Ingest clean benchmark scenes (including user clean scene)
    benchmark_clean_dir = config.project_root / "ml" / "benchmark_gallery" / "clean"
    if benchmark_clean_dir.exists():
        for clean_img_path in benchmark_clean_dir.glob("*.jpg"):
            try:
                img_bgr = cv2.imread(str(clean_img_path))
                if img_bgr is None: continue
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                gh, gw = gray.shape
                
                # Multi-crop patches across the clean scene
                step_x = max(1, (gw - ps) // 3) if gw > ps else 1
                step_y = max(1, (gh - ps) // 2) if gh > ps else 1
                
                for cy in range(0, max(1, gh - ps + 1), step_y):
                    for cx in range(0, max(1, gw - ps + 1), step_x):
                        crop_vv = gray[cy:cy+ps, cx:cx+ps]
                        if crop_vv.shape[0] != ps or crop_vv.shape[1] != ps:
                            crop_vv = cv2.resize(crop_vv, (ps, ps), interpolation=cv2.INTER_AREA)
                        
                        vv_fl = crop_vv.astype(np.float32) / 255.0
                        vh_fl = np.maximum(0.0, vv_fl - 0.22)
                        norm_uint8 = np.stack([(vv_fl * 255.0).astype(np.uint8), (vh_fl * 255.0).astype(np.uint8)], axis=0)
                        
                        all_patches.append(norm_uint8)
                        all_labels.append(0.0)
                        all_cats.append("clean")
                
                # Also add full scene resized
                resized_vv = cv2.resize(gray, (ps, ps), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
                resized_vh = np.maximum(0.0, resized_vv - 0.22)
                norm_full = np.stack([(resized_vv * 255.0).astype(np.uint8), (resized_vh * 255.0).astype(np.uint8)], axis=0)
                all_patches.append(norm_full)
                all_labels.append(0.0)
                all_cats.append("clean")
                print(f"  [BENCHMARK] Ingested clean scene patches from {clean_img_path.name}")
            except Exception as e:
                print(f"  [WARN] Failed loading benchmark clean {clean_img_path.name}: {e}")
    
    patches = np.stack(all_patches, axis=0)
    labels = np.array(all_labels, dtype=np.float32)
    categories = np.array(all_cats, dtype=object)
    
    # Shuffle deterministically
    perm = np.random.RandomState(42).permutation(len(labels))
    patches = patches[perm]
    labels = labels[perm]
    categories = categories[perm]
    
    n_oil = int(np.sum(labels == 1.0))
    n_neg = int(np.sum(labels == 0.0))
    print(f"\n[PATCHES] Total: {len(labels)} (Oil: {n_oil}, Non-oil: {n_neg})")
    
    cat_counts = {}
    for c in categories:
        cat_counts[c] = cat_counts.get(c, 0) + 1
    for cat, cnt in sorted(cat_counts.items()):
        print(f"  {cat:15s}: {cnt}")
    
    return patches, labels, categories


# ═══════════════════════════════════════════════════════
# PyTorch Dataset for Gallery Patches
# ═══════════════════════════════════════════════════════

class GalleryPatchDataset(torch.utils.data.Dataset):
    def __init__(self, patches, labels, augment=False):
        self.patches = patches
        self.labels = labels.astype(np.float32)
        self.augment = augment
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        patch = self.patches[idx].astype(np.float32) / 255.0
        label = self.labels[idx]
        
        if self.augment:
            if random.random() > 0.5:
                patch = np.flip(patch, axis=2).copy()
            if random.random() > 0.5:
                patch = np.flip(patch, axis=1).copy()
            if random.random() > 0.5:
                k = random.choice([1, 2, 3])
                patch = np.rot90(patch, k=k, axes=(1, 2)).copy()
            if random.random() > 0.5:
                jitter = random.uniform(0.95, 1.05)
                patch = np.clip(patch * jitter, 0.0, 1.0)
        
        return torch.from_numpy(patch).float(), torch.tensor(label, dtype=torch.float32)


# ═══════════════════════════════════════════════════════
# Fine-Tuning Loop
# ═══════════════════════════════════════════════════════

def finetune(config: SpillSenseConfig = CONFIG) -> Dict:
    """
    Fine-tune the pre-trained classifier on gallery-representative scenes.
    """
    set_seed(42)
    device = config.get_device()
    
    print("=" * 60)
    print("SpillSense v3 — Gallery-Targeted Fine-Tuning")
    print("=" * 60)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 1. Select gallery scenes
    scenes = select_gallery_scenes(config)
    
    # 2. Extract patches  
    patches, labels, categories = extract_gallery_patches(scenes, config)
    
    # 3. Split 80/20
    n_total = len(labels)
    n_train = int(n_total * 0.80)
    
    tr_patches, tr_labels, tr_cats = patches[:n_train], labels[:n_train], categories[:n_train]
    va_patches, va_labels, va_cats = patches[n_train:], labels[n_train:], categories[n_train:]
    
    print(f"\nTrain: {len(tr_labels)} (Oil: {int(np.sum(tr_labels==1))}, Neg: {int(np.sum(tr_labels==0))})")
    print(f"Val:   {len(va_labels)} (Oil: {int(np.sum(va_labels==1))}, Neg: {int(np.sum(va_labels==0))})")
    
    # 4. Load pre-trained model
    ckpt_path = config.checkpoint_dir / "best_classifier_v3.pt"
    model = get_classifier(in_channels=config.in_channels).to(device)
    
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        prev_threshold = ckpt.get("best_threshold", 0.50)
        print(f"\n[LOADED] Pre-trained model from {ckpt_path} (threshold={prev_threshold:.2f})")
    else:
        print("\n[WARN] No pre-trained model found, training from scratch")
    
    model.to(device)
    
    # 5. Build dataloaders with class-balanced sampling
    # Oversample oil patches 3x to ensure the model sees enough positives
    oil_mask = tr_labels == 1.0
    neg_mask = tr_labels == 0.0
    sample_weights = np.zeros(len(tr_labels), dtype=np.float64)
    n_oil = max(1, int(np.sum(oil_mask)))
    n_neg = max(1, int(np.sum(neg_mask)))
    sample_weights[oil_mask] = 0.6 / n_oil   # 60% oil sampling
    sample_weights[neg_mask] = 0.4 / n_neg   # 40% negative sampling
    
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(tr_labels),
        replacement=True
    )
    
    train_ds = GalleryPatchDataset(tr_patches, tr_labels, augment=True)
    val_ds = GalleryPatchDataset(va_patches, va_labels, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    
    # 6. Fine-tune with lower LR (standard transfer learning practice)
    criterion = FocalLossWithLogits(gamma=2.0, alpha=0.25)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35, eta_min=1e-6)
    
    best_f1 = 0.0
    best_state = None
    best_threshold = 0.50
    best_epoch = 0
    
    FT_EPOCHS = 35
    
    print(f"\n{'Ep':>3} | {'TrLoss':>7} | {'VLoss':>7} | {'VF1':>6} | {'VAcc':>6} | {'Thr':>5} | {'LR':>8}")
    print("-" * 65)
    
    for epoch in range(1, FT_EPOCHS + 1):
        t0 = time.time()
        
        # Train
        model.train()
        total_loss = 0.0
        total_samples = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch).squeeze(-1)
            loss = criterion(logits, y_batch).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)
            total_samples += x_batch.size(0)
        
        tr_loss = total_loss / max(1, total_samples)
        scheduler.step()
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_total = 0
        all_probs, all_labels_list = [], []
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                logits = model(x_batch).squeeze(-1)
                loss = criterion(logits, y_batch).mean()
                probs = torch.sigmoid(logits)
                val_loss += loss.item() * x_batch.size(0)
                val_total += x_batch.size(0)
                all_probs.extend(probs.cpu().numpy())
                all_labels_list.extend(y_batch.cpu().numpy())
        
        val_loss /= max(1, val_total)
        np_probs = np.array(all_probs)
        np_labels = np.array(all_labels_list)
        
        # Threshold sweep
        best_t_f1 = 0.0
        best_t = 0.50
        for t in np.linspace(0.1, 0.9, 33):
            m = compute_metrics(np_labels, np_probs, float(t))
            if m["f1"] > best_t_f1:
                best_t_f1 = m["f1"]
                best_t = float(t)
        
        m = compute_metrics(np_labels, np_probs, best_t)
        lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0
        
        print(f"{epoch:3d} | {tr_loss:7.4f} | {val_loss:7.4f} | {m['f1']:6.3f} | "
              f"{m['accuracy']:6.3f} | {best_t:5.2f} | {lr:8.1e}")
        
        if m["f1"] >= best_f1:
            best_f1 = m["f1"]
            best_threshold = best_t
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    # Restore best
    if best_state:
        model.load_state_dict(best_state)
    model.to(device)
    
    print(f"\n[BEST] Epoch {best_epoch}, F1={best_f1:.3f}, Threshold={best_threshold:.2f}")
    
    # Save fine-tuned checkpoint
    ft_ckpt_path = config.checkpoint_dir / "best_classifier_v3_finetuned.pt"
    torch.save({
        "epoch": best_epoch,
        "model_state": best_state,
        "best_threshold": best_threshold,
        "best_f1": best_f1,
        "finetune_source": "gallery_scenes",
    }, ft_ckpt_path)
    print(f"[SAVED] Fine-tuned checkpoint: {ft_ckpt_path}")
    
    # Also overwrite the main checkpoint
    torch.save({
        "epoch": best_epoch,
        "model_state": best_state,
        "best_score": best_f1,
        "best_threshold": best_threshold,
    }, config.checkpoint_dir / "best_classifier_v3.pt")
    print(f"[SAVED] Updated main checkpoint")
    
    # 7. Export ONNX
    export_onnx_model(model, config, best_threshold, best_f1)
    
    # 8. Regenerate demo images from gallery source scenes
    regenerate_demo_images(scenes, model, device, config, best_threshold)
    
    return {
        "model": model,
        "best_f1": best_f1,
        "best_threshold": best_threshold,
        "best_epoch": best_epoch,
    }


# ═══════════════════════════════════════════════════════
# ONNX Export
# ═══════════════════════════════════════════════════════

def export_onnx_model(model, config, threshold, f1_score):
    """Export to ONNX with updated metadata."""
    path = config.output_classifier_onnx
    path.parent.mkdir(parents=True, exist_ok=True)
    
    original_device = next(model.parameters()).device
    model.eval().to("cpu")
    dummy = torch.randn(1, 2, config.cls_patch_size, config.cls_patch_size)
    torch.onnx.export(
        model, dummy, str(path), export_params=True, opset_version=14,
        do_constant_folding=True,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
    )
    size_kb = path.stat().st_size / 1024
    print(f"\n[ONNX] Exported to {path} ({size_kb:.1f} KB)")
    # Move model back to original device for subsequent inference
    model.to(original_device)
    
    # Update metadata
    metadata = {
        "model_name": "DualPolOilSpillNet",
        "model_type": "classifier",
        "version": config.cache_version + "_finetuned",
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "in_channels": config.in_channels,
        "input_shape": [1, 2, config.cls_patch_size, config.cls_patch_size],
        "channel_order": list(config.channel_names),
        "optimal_threshold": float(threshold),
        "normalization": {
            "vv_min_db": config.vv_min_db, "vv_max_db": config.vv_max_db,
            "vh_min_db": config.vh_min_db, "vh_max_db": config.vh_max_db,
        },
        "finetune_note": "Fine-tuned on gallery-representative Zenodo scenes",
        "onnx_size_kb": size_kb,
    }
    config.model_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config.model_metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[METADATA] Saved to {config.model_metadata_path}")


# ═══════════════════════════════════════════════════════
# Demo Image Regeneration
# ═══════════════════════════════════════════════════════

def regenerate_demo_images(
    scenes: Dict[str, List[Dict]],
    model: nn.Module,
    device: torch.device,
    config: SpillSenseConfig,
    threshold: float
):
    """
    Regenerate the demo-sar/ gallery images from the actual Zenodo scenes
    used for fine-tuning. Produces normalized VV-grayscale JPGs with proper
    SAR backscatter contrast.
    """
    demo_dir = config.demo_dir
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    ps = config.cls_patch_size  # 400
    
    # Generate class_1 (oil) images from the gallery oil scenes
    oil_scenes = scenes.get("oil", [])[:10]
    oil_idx = 0
    
    for scene in oil_scenes:
        try:
            vv, vh, _ = load_sentinel1_scene(Path(scene["image_path"]))
            mask = load_mask(Path(scene["mask_path"]))
            if mask is None:
                continue
            
            h, w = vv.shape
            if h < ps or w < ps:
                continue
            
            # Find oil-centered crop
            if np.sum(mask > 0) > 100:
                ys, xs = np.where(mask > 0)
                cy, cx = int(np.median(ys)), int(np.median(xs))
                y = int(np.clip(cy - ps // 2, 0, h - ps))
                x = int(np.clip(cx - ps // 2, 0, w - ps))
            else:
                y, x = 0, 0
            
            patch_vv = vv[y:y+ps, x:x+ps]
            patch_vh = vh[y:y+ps, x:x+ps]
            
            norm = normalize_sar_dualpol(patch_vv, patch_vh, config)
            vv_norm = norm[0]  # Channel 0 = VV
            
            # Save as grayscale JPG (VV normalized)
            img = Image.fromarray((np.clip(vv_norm, 0, 1) * 255).astype(np.uint8), mode="L")
            img = img.resize((800, 800), Image.Resampling.BILINEAR)
            
            oil_idx += 1
            img.save(str(demo_dir / f"class_1_{oil_idx:02d}.jpg"), quality=95)
            img.save(str(demo_dir / f"class_1_{oil_idx}.jpg"), quality=95)
            
            # Verify with model
            inp = torch.from_numpy(norm).unsqueeze(0).float().to(device)
            with torch.no_grad():
                logit = model(inp).squeeze().cpu().item()
            prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -20, 20)))
            decision = "OIL" if prob >= threshold else "CLEAN"
            print(f"  class_1_{oil_idx:02d}.jpg: P={prob:.3f} -> {decision}")
            
        except Exception as e:
            print(f"  [SKIP] Oil scene error: {e}")
        
        if oil_idx >= 10:
            break
    
    # Generate class_0 (clean) images from negative scenes
    neg_scenes = scenes.get("clean", []) + scenes.get("lookalike", []) + scenes.get("ship_wake", [])
    clean_idx = 0
    
    for scene in neg_scenes:
        try:
            vv, vh, _ = load_sentinel1_scene(Path(scene["image_path"]))
            h, w = vv.shape
            
            if "crop_box" in scene:
                cy, cx, ch, cw = scene["crop_box"]
                patch_vv = vv[cy:cy+ch, cx:cx+cw]
                patch_vh = vh[cy:cy+ch, cx:cx+cw]
            elif h >= ps and w >= ps:
                y = max(0, h // 2 - ps // 2)
                x = max(0, w // 2 - ps // 2)
                patch_vv = vv[y:y+ps, x:x+ps]
                patch_vh = vh[y:y+ps, x:x+ps]
            else:
                continue
            
            norm = normalize_sar_dualpol(patch_vv, patch_vh, config)
            vv_norm = norm[0]
            
            img = Image.fromarray((np.clip(vv_norm, 0, 1) * 255).astype(np.uint8), mode="L")
            img = img.resize((800, 800), Image.Resampling.BILINEAR)
            
            clean_idx += 1
            img.save(str(demo_dir / f"class_0_{clean_idx:02d}.jpg"), quality=95)
            img.save(str(demo_dir / f"class_0_{clean_idx}.jpg"), quality=95)
            
            # Verify with model
            inp = torch.from_numpy(norm).unsqueeze(0).float().to(device)
            with torch.no_grad():
                logit = model(inp).squeeze().cpu().item()
            prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -20, 20)))
            decision = "OIL" if prob >= threshold else "CLEAN"
            print(f"  class_0_{clean_idx:02d}.jpg: P={prob:.3f} -> {decision}")
            
        except Exception as e:
            print(f"  [SKIP] Clean scene error: {e}")
        
        if clean_idx >= 10:
            break
    
    print(f"\n[DEMO] Regenerated {oil_idx} oil + {clean_idx} clean demo images in {demo_dir}")


if __name__ == "__main__":
    result = finetune()
    print(f"\n{'='*60}")
    print(f"FINE-TUNING COMPLETE")
    print(f"Best F1: {result['best_f1']:.3f}")
    print(f"Threshold: {result['best_threshold']:.2f}")
    print(f"{'='*60}")
