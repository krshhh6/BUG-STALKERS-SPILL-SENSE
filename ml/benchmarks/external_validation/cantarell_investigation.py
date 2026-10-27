"""
SpillSense - Forensic Investigation of Cantarell Domain Shift & False Positives
================================================================================
Investigates the discrepancy between uncalibrated 8-bit RGB rendered tiles vs.
calibrated 16-bit radiometric Sigma0 radar backscatter (dB).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import json
import torch
import numpy as np
import pandas as pd
from ml.models import get_segmenter
from ml.config import CONFIG

def run_cantarell_forensic():
    print("=" * 80)
    print("CANTARELL DOMAIN SHIFT & FALSE-POSITIVE FORENSIC INVESTIGATION")
    print("=" * 80)

    ckpt_path = CONFIG.checkpoint_dir / "best_segmenter_v3_exp1_lr1e4.pt"
    model = get_segmenter(in_channels=2)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_csv = ROOT / "ml/benchmarks/external_validation/datasets/D_Cantarell/repo/splits/test.csv"
    if not test_csv.exists():
        print(f"[ERROR] test.csv not found at {test_csv}")
        return

    df = pd.read_csv(test_csv)
    base_dir = test_csv.parent.parent

    records = []

    for idx, row in df.iterrows():
        sample_id = row["sample_id"]
        rgb_path = base_dir / f"IMAGES/IMG-RGB/{sample_id}.png"
        tiff_path = base_dir / f"RASTER/IMG-TIFF/16-BIT/{sample_id}.tiff"
        mask_path = base_dir / f"IMAGES/LABELS-1D/{sample_id}_1D.png"

        if not (rgb_path.exists() and tiff_path.exists() and mask_path.exists()):
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8)
        gt_oil = int(np.sum(mask == 1))

        # --- Mode 1: 8-Bit RGB uncalibrated (naive img / 255.0) ---
        img_8bit = cv2.imread(str(rgb_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        vh_8bit = np.clip(img_8bit - 0.20, 0.0, 1.0)
        inp_8bit = torch.from_numpy(np.stack([img_8bit, vh_8bit], axis=0)).unsqueeze(0).float()
        inp_8bit = torch.nn.functional.interpolate(inp_8bit, size=(512, 512), mode="bilinear")
        with torch.no_grad():
            prob_8bit = torch.sigmoid(model(inp_8bit)).squeeze().numpy()
        prob_8bit_full = cv2.resize(prob_8bit, (800, 600))
        pred_8bit = (prob_8bit_full > 0.5).astype(np.uint8)

        tp_8 = int(np.sum((pred_8bit == 1) & (mask == 1)))
        fp_8 = int(np.sum((pred_8bit == 1) & (mask == 0)))
        fn_8 = int(np.sum((pred_8bit == 0) & (mask == 1)))
        dice_8 = 2*tp_8 / (2*tp_8 + fp_8 + fn_8) if (2*tp_8 + fp_8 + fn_8) > 0 else 0.0
        prec_8 = tp_8 / (tp_8 + fp_8) if (tp_8 + fp_8) > 0 else 0.0
        rec_8 = tp_8 / (tp_8 + fn_8) if (tp_8 + fn_8) > 0 else 0.0

        # --- Mode 2: 16-Bit Calibrated Sigma0 dB (radiometric normalization) ---
        tiff = cv2.imread(str(tiff_path), cv2.IMREAD_UNCHANGED)
        vv_norm = np.clip((tiff + 35.0) / 40.0, 0.0, 1.0)
        vh_db = tiff - 9.0  # Physical cross-polarization Bragg depression
        vh_norm = np.clip((vh_db + 35.0) / 40.0, 0.0, 1.0)
        inp_16 = torch.from_numpy(np.stack([vv_norm, vh_norm], axis=0)).unsqueeze(0).float()
        inp_16 = torch.nn.functional.interpolate(inp_16, size=(512, 512), mode="bilinear")
        with torch.no_grad():
            prob_16 = torch.sigmoid(model(inp_16)).squeeze().numpy()
        prob_16_full = cv2.resize(prob_16, (800, 600))
        pred_16 = (prob_16_full > 0.5).astype(np.uint8)

        tp_16 = int(np.sum((pred_16 == 1) & (mask == 1)))
        fp_16 = int(np.sum((pred_16 == 1) & (mask == 0)))
        fn_16 = int(np.sum((pred_16 == 0) & (mask == 1)))
        dice_16 = 2*tp_16 / (2*tp_16 + fp_16 + fn_16) if (2*tp_16 + fp_16 + fn_16) > 0 else 0.0
        prec_16 = tp_16 / (tp_16 + fp_16) if (tp_16 + fp_16) > 0 else 0.0
        rec_16 = tp_16 / (tp_16 + fn_16) if (tp_16 + fn_16) > 0 else 0.0

        records.append({
            "sample_id": sample_id,
            "gt_oil_pixels": gt_oil,
            "8bit_tp": tp_8, "8bit_fp": fp_8, "8bit_fn": fn_8, "8bit_dice": dice_8, "8bit_prec": prec_8, "8bit_rec": rec_8,
            "16bit_tp": tp_16, "16bit_fp": fp_16, "16bit_fn": fn_16, "16bit_dice": dice_16, "16bit_prec": prec_16, "16bit_rec": rec_16
        })

    rdf = pd.DataFrame(records)
    out_csv = ROOT / "ml/benchmarks/external_validation/cantarell_forensic_comparison.csv"
    rdf.to_csv(out_csv, index=False)

    # Micro aggregates
    tot_tp8, tot_fp8, tot_fn8 = rdf["8bit_tp"].sum(), rdf["8bit_fp"].sum(), rdf["8bit_fn"].sum()
    tot_tp16, tot_fp16, tot_fn16 = rdf["16bit_tp"].sum(), rdf["16bit_fp"].sum(), rdf["16bit_fn"].sum()

    micro_dice8 = 2*tot_tp8 / (2*tot_tp8 + tot_fp8 + tot_fn8)
    micro_prec8 = tot_tp8 / (tot_tp8 + tot_fp8)
    micro_rec8 = tot_tp8 / (tot_tp8 + tot_fn8)
    micro_iou8 = tot_tp8 / (tot_tp8 + tot_fp8 + tot_fn8)

    micro_dice16 = 2*tot_tp16 / (2*tot_tp16 + tot_fp16 + tot_fn16)
    micro_prec16 = tot_tp16 / (tot_tp16 + tot_fp16)
    micro_rec16 = tot_tp16 / (tot_tp16 + tot_fn16)
    micro_iou16 = tot_tp16 / (tot_tp16 + tot_fp16 + tot_fn16)

    print("\n" + "=" * 80)
    print("FORENSIC RESULTS (22 TEST TILES - CANTARELL GULF OF MEXICO)")
    print("=" * 80)
    print(f"{'Metric':<25} | {'8-bit Uncalibrated RGB':<22} | {'16-bit Calibrated Sigma0 dB':<25}")
    print("-" * 78)
    print(f"{'Micro Dice':<25} | {micro_dice8:<22.4f} | {micro_dice16:<25.4f}")
    print(f"{'Micro IoU (Jaccard)':<25} | {micro_iou8:<22.4f} | {micro_iou16:<25.4f}")
    print(f"{'Micro Precision':<25} | {micro_prec8:<22.4f} | {micro_prec16:<25.4f}")
    print(f"{'Micro Recall':<25} | {micro_rec8:<22.4f} | {micro_rec16:<25.4f}")
    print(f"{'Total False Positives':<25} | {tot_fp8:<22,d} | {tot_fp16:<25,d}")
    print(f"{'False Positive Reduction':<25} | {'Reference':<22} | {((tot_fp8 - tot_fp16) / tot_fp8 * 100):.2f}% Reduction")
    print("=" * 80)

    # Save JSON summary
    summary = {
        "dataset": "Cantarell (Test Split, 22 tiles)",
        "uncalibrated_8bit": {
            "micro_dice": float(micro_dice8),
            "micro_iou": float(micro_iou8),
            "micro_precision": float(micro_prec8),
            "micro_recall": float(micro_rec8),
            "false_positives": int(tot_fp8)
        },
        "calibrated_16bit_sigma0_db": {
            "micro_dice": float(micro_dice16),
            "micro_iou": float(micro_iou16),
            "micro_precision": float(micro_prec16),
            "micro_recall": float(micro_rec16),
            "false_positives": int(tot_fp16),
            "fp_reduction_percent": float((tot_fp8 - tot_fp16) / tot_fp8 * 100)
        }
    }
    with open(ROOT / "ml/benchmarks/external_validation/cantarell_forensic_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    run_cantarell_forensic()
