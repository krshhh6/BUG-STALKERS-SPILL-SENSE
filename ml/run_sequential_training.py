"""
Sequential training continuation runner for SpillSense v3.
Executes sequentially on GPU:
1. DualPolOilSpillNet classifier (--resume, --out-suffix _exp3_precision)
2. SpillSegNet segmenter (--resume, --out-suffix _exp2_precision, --max-scenes 1200)
"""
import sys
import subprocess
from pathlib import Path

def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    root = Path(__file__).resolve().parent.parent

    print("=" * 70)
    print("SpillSense v3 — Sequential Training Continuation")
    print("Step 1/2: DualPolOilSpillNet Classifier (_exp3_precision, --resume)")
    print("Step 2/2: SpillSegNet Segmenter (_exp2_precision, --max-scenes 1200, --resume)")
    print("=" * 70, flush=True)

    cls_cmd = [
        sys.executable, "-u", "ml/train.py",
        "--epochs", "35",
        "--patience", "7",
        "--out-suffix", "_exp3_precision",
        "--seed", "42",
        "--resume"
    ]
    print(f"\n[RUNNING STEP 1] {' '.join(cls_cmd)}\n", flush=True)
    res1 = subprocess.run(cls_cmd, cwd=root)
    print(f"\n[STEP 1 FINISHED] Exit code: {res1.returncode}\n", flush=True)

    seg_cmd = [
        sys.executable, "-u", "ml/train_segmentation.py",
        "--max-scenes", "1200",
        "--lr", "1e-4",
        "--epochs", "30",
        "--patience", "6",
        "--out-suffix", "_exp2_precision",
        "--seed", "42",
        "--resume"
    ]
    print(f"\n[RUNNING STEP 2] {' '.join(seg_cmd)}\n", flush=True)
    res2 = subprocess.run(seg_cmd, cwd=root)
    print(f"\n[STEP 2 FINISHED] Exit code: {res2.returncode}\n", flush=True)

    print("=" * 70)
    print("Sequential Training Continuation Finished!")
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
