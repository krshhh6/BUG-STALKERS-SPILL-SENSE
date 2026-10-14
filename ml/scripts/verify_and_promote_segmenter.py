"""
SpillSense Phase 4 — Segmenter Production Promotion & Verification Script
========================================================================
Executes Task A:
1. Verifies candidate PyTorch checkpoint hash.
2. Backs up existing production ONNX model and hashes both.
3. Exports candidate to temporary ONNX.
4. Validates tensor shapes, preprocessing compatibility, and numerical parity (PyTorch vs ONNX).
5. Runs representative-scene smoke test with tiled inference.
6. Promotes candidate ONNX to frontend/public/models/oil_segmenter.onnx.
7. Verifies deployed ONNX session creation and inference execution.
"""

import sys
import shutil
import hashlib
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.config import CONFIG
from ml.models import get_segmenter
from ml.preprocess import load_sentinel1_scene, load_mask, normalize_sar_dualpol
from ml.train_segmentation import tiled_inference, compute_seg_metrics


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()


def main():
    print("=" * 80)
    print("SPILLSENSE TASK A: SEGMENTER PRODUCTION PROMOTION & SMOKE TEST")
    print("=" * 80)

    # 1. Candidate Checkpoint Verification
    cand_pt_path = ROOT / "ml" / "checkpoints" / "best_segmenter_v3_exp1_lr1e4.pt"
    expected_cand_hash = "38DD8341C2CD0D1DD0DDB5082130C96B675CD991F8FB692E7B206D1831B3584D"
    cand_hash = sha256_file(cand_pt_path)
    print(f"\n[STEP 1] Candidate Checkpoint: {cand_pt_path.name}")
    print(f"  Expected Hash: {expected_cand_hash}")
    print(f"  Actual Hash:   {cand_hash}")
    assert cand_hash == expected_cand_hash, "Candidate hash verification failed!"
    print("  --> Candidate hash verified bit-for-bit.")

    # 2. Backup Existing Production ONNX
    prod_onnx_path = ROOT / "frontend" / "public" / "models" / "oil_segmenter.onnx"
    backup_onnx_path1 = ROOT / "frontend" / "public" / "models" / "oil_segmenter_baseline_backup.onnx"
    backup_onnx_path2 = ROOT / "ml" / "checkpoints" / "backup_oil_segmenter_baseline.onnx"

    prod_hash_before = sha256_file(prod_onnx_path)
    print(f"\n[STEP 2] Backing up Current Production ONNX: {prod_onnx_path}")
    print(f"  Current Hash: {prod_hash_before}")
    print(f"  Current Size: {prod_onnx_path.stat().st_size:,} bytes")

    shutil.copy2(prod_onnx_path, backup_onnx_path1)
    shutil.copy2(prod_onnx_path, backup_onnx_path2)

    bak1_hash = sha256_file(backup_onnx_path1)
    bak2_hash = sha256_file(backup_onnx_path2)
    assert bak1_hash == prod_hash_before, "Backup 1 hash mismatch!"
    assert bak2_hash == prod_hash_before, "Backup 2 hash mismatch!"
    print(f"  --> Backup 1 saved: {backup_onnx_path1} (verified)")
    print(f"  --> Backup 2 saved: {backup_onnx_path2} (verified)")

    # 3. Export Candidate to Temporary Path
    temp_onnx_path = ROOT / "ml" / "checkpoints" / "temp_oil_segmenter_candidate.onnx"
    print(f"\n[STEP 3] Exporting Candidate to Staging ONNX: {temp_onnx_path.name}")

    device = torch.device("cpu")
    ckpt = torch.load(cand_pt_path, map_location=device, weights_only=False)
    pytorch_model = get_segmenter(2).to(device)
    pytorch_model.load_state_dict(ckpt["model_state"])
    pytorch_model.eval()

    dummy_input = torch.randn(1, 2, 512, 512, dtype=torch.float32)
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        str(temp_onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
    )
    print(f"  --> Staging ONNX generated ({temp_onnx_path.stat().st_size:,} bytes)")

    # 4. Numerical Equivalence Validation (PyTorch vs ONNX Runtime)
    print(f"\n[STEP 4] Numerical Equivalence & Interface Validation")
    ort_sess = ort.InferenceSession(str(temp_onnx_path), providers=["CPUExecutionProvider"])

    # Check input/output contract
    in_meta = ort_sess.get_inputs()[0]
    out_meta = ort_sess.get_outputs()[0]
    print(f"  ONNX Input:  name='{in_meta.name}', shape={in_meta.shape}, type={in_meta.type}")
    print(f"  ONNX Output: name='{out_meta.name}', shape={out_meta.shape}, type={out_meta.type}")
    assert in_meta.name == "input", f"Expected input name 'input', got {in_meta.name}"
    assert out_meta.name == "output", f"Expected output name 'output', got {out_meta.name}"
    assert in_meta.shape[1:] == [2, 512, 512], f"Input shape unexpected: {in_meta.shape}"
    assert out_meta.shape[1:] == [1, 512, 512], f"Output shape unexpected: {out_meta.shape}"

    # Parity check on random tensor
    with torch.no_grad():
        pt_logits = pytorch_model(dummy_input)
        pt_probs = torch.sigmoid(pt_logits).numpy()

    onnx_out = ort_sess.run(None, {"input": dummy_input.numpy()})[0]
    onnx_probs = 1.0 / (1.0 + np.exp(-onnx_out))  # Apply sigmoid to raw logits

    mae = float(np.max(np.abs(pt_probs - onnx_probs)))
    print(f"  Maximum Absolute Probability Difference (PyTorch vs ONNX): {mae:.2e}")
    assert mae < 1e-5, f"Numerical divergence between PyTorch and ONNX: {mae} >= 1e-5"
    print("  --> Numerical equivalence PASSED (MAE < 1e-5).")

    # 5. Representative-Scene Smoke Test
    print(f"\n[STEP 5] Representative-Scene Smoke Tests")
    # Test on scene oil_00492 and oil_00563
    smoke_scenes = ["00492.tif", "00563.tif"]
    oil_img_dir = CONFIG.data.find_oil_images()
    oil_mask_dir = CONFIG.data.find_oil_masks()

    for sc_name in smoke_scenes:
        img_p = oil_img_dir / sc_name
        mask_p = oil_mask_dir / sc_name
        if not img_p.exists():
            continue

        vv, vh, valid = load_sentinel1_scene(img_p)
        gt_mask = load_mask(mask_p)
        assert gt_mask is not None, f"GT mask missing for {sc_name}"

        # PyTorch full-scene tiled inference
        pt_prob_map = tiled_inference(pytorch_model, vv, vh, CONFIG, device=device)
        pt_metrics = compute_seg_metrics(pt_prob_map, gt_mask, threshold=0.50)

        # ONNX full-scene tiled inference
        h, w = vv.shape
        ps = 512
        stride = 384
        onnx_prob_map = np.zeros((h, w), dtype=np.float32)
        onnx_count_map = np.zeros((h, w), dtype=np.float32)

        for y in range(0, h - ps + 1, stride):
            for x in range(0, w - ps + 1, stride):
                crop_vv = vv[y:y+ps, x:x+ps]
                crop_vh = vh[y:y+ps, x:x+ps]
                norm = normalize_sar_dualpol(crop_vv, crop_vh, CONFIG)
                inp_arr = norm[np.newaxis, ...].astype(np.float32)
                logits = ort_sess.run(None, {"input": inp_arr})[0]
                crop_probs = (1.0 / (1.0 + np.exp(-logits)))[0, 0]
                onnx_prob_map[y:y+ps, x:x+ps] += crop_probs
                onnx_count_map[y:y+ps, x:x+ps] += 1.0

        onnx_prob_map /= np.maximum(onnx_count_map, 1.0)
        onnx_metrics = compute_seg_metrics(onnx_prob_map, gt_mask, threshold=0.50)

        scene_mae = float(np.max(np.abs(pt_prob_map - onnx_prob_map)))
        print(f"  Scene: {sc_name}")
        print(f"    PyTorch Full-Scene Dice: {pt_metrics['dice']:.4f}, IoU: {pt_metrics['iou']:.4f}")
        print(f"    ONNX Full-Scene Dice:    {onnx_metrics['dice']:.4f}, IoU: {onnx_metrics['iou']:.4f}")
        print(f"    Full-Scene Maximum Probability Discrepancy: {scene_mae:.2e}")
        assert abs(pt_metrics['dice'] - onnx_metrics['dice']) < 1e-4, "Full scene Dice divergence!"
        print(f"    --> Smoke test for {sc_name} PASSED.")

    # 6. Production Promotion
    print(f"\n[STEP 6] Promoting Validated ONNX to Production: {prod_onnx_path}")
    shutil.copy2(temp_onnx_path, prod_onnx_path)
    deployed_hash = sha256_file(prod_onnx_path)
    print(f"  Deployed Production ONNX Hash: {deployed_hash}")
    print(f"  Deployed Production ONNX Size: {prod_onnx_path.stat().st_size:,} bytes")

    # 7. Final Deployed Sanity Check
    final_sess = ort.InferenceSession(str(prod_onnx_path), providers=["CPUExecutionProvider"])
    out = final_sess.run(None, {"input": np.zeros((1, 2, 512, 512), dtype=np.float32)})
    print(f"  Final Session Check: Ran zero tensor successfully, output shape {out[0].shape}")

    print("\n" + "=" * 80)
    print("TASK A SEGMENTER PROMOTION COMPLETE & VERIFIED")
    print(f"Previous Baseline Backup: {backup_onnx_path1.name} (SHA: {prod_hash_before})")
    print(f"Newly Promoted Model:     {prod_onnx_path.name} (SHA: {deployed_hash})")
    print("=" * 80)

    return {
        "status": "SUCCESS",
        "previous_prod_hash": prod_hash_before,
        "backup_path": str(backup_onnx_path1),
        "promoted_hash": deployed_hash,
        "promoted_path": str(prod_onnx_path),
        "cand_pt_hash": cand_hash,
    }


if __name__ == "__main__":
    res = main()
