"""
SpillSense External Validation - Dataset Downloader & Preparer
================================================================
Downloads, extracts, and inventories external benchmark datasets.
"""

import os
import sys
import csv
import json
import time
import hashlib
import zipfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import requests
except ImportError:
    requests = None

DATASETS_ROOT = Path(__file__).resolve().parent / "datasets"
INVENTORY_PATH = Path(__file__).resolve().parent / "dataset_inventory.csv"

DATASET_REGISTRY = {
    "A_SOS": {
        "name": "Refined Deep-SAR Oil Spill (SOS)",
        "source_url": "https://zenodo.org/records/15298010",
        "files": [
            {
                "url": "https://zenodo.org/records/15298010/files/images.zip",
                "filename": "images.zip",
                "expected_md5": "e5272875611e8b5a2a0d49972224c842",
                "size_bytes": 1148063776,
            },
            {
                "url": "https://zenodo.org/records/15298010/files/masks.zip",
                "filename": "masks.zip",
                "expected_md5": "09c8a279603c8d7b73c6dc64f91e1106",
                "size_bytes": 28276974,
            },
        ],
        "license": "CC BY 4.0 (inferred from Zenodo)",
        "task": "segmentation",
        "polarization": "single (VV or intensity)",
        "notes": "Refined Deep-SAR SOS. Single-pol SAR patches with binary masks.",
    },
    "B_DARTIS": {
        "name": "DARTIS - Eastern Mediterranean Oil Slicks 2019",
        "source_url": "https://doi.pangaea.de/10.1594/PANGAEA.980773",
        "files": [],
        "license": "CC BY 4.0",
        "task": "classification / detection",
        "polarization": "metadata-only (tabular)",
        "notes": "Tabular metadata dataset. Requires Copernicus API to fetch SAR images.",
        "blocker": "DARTIS provides annotations/metadata only, not pre-cropped images.",
    },
    "C_CSIRO": {
        "name": "CSIRO Sentinel-1 SAR Data",
        "source_url": "https://data.csiro.au/",
        "files": [],
        "license": "Unknown",
        "task": "variable",
        "polarization": "unknown",
        "notes": "No specific oil spill collection with segmentation masks identified.",
        "blocker": "No specific CSIRO SAR oil spill benchmark dataset identified.",
    },
    "D_Cantarell": {
        "name": "Cantarell Oil Spill/Seepage Segmentation Dataset",
        "source_url": "https://github.com/Garcia-INPE/oilspill-detection-dataset",
        "files": [
            {
                "url": "https://github.com/Garcia-INPE/oilspill-detection-dataset/archive/refs/heads/main.zip",
                "filename": "cantarell_dataset.zip",
                "expected_md5": None,
                "size_bytes": None,
            },
        ],
        "license": "CC BY 4.0",
        "task": "segmentation",
        "polarization": "single (derived RGB from SAR)",
        "notes": "173 image-mask triplets (~800x600 px). Gulf of Mexico / Cantarell.",
    },
    "E_Zenodo_Sentinel1": {
        "name": "Sentinel-1 SAR Oil Spill Dataset (Trujillo-Acatitla et al.)",
        "source_url": "https://zenodo.org/records/8346860",
        "files": [],
        "license": "CC BY 4.0",
        "task": "segmentation + classification",
        "polarization": "dual (VV + VH)",
        "notes": "THIS IS OUR TRAINING DATASET - not valid for external evaluation.",
        "blocker": "Dataset E is the same dataset used to train our models.",
    },
}


def check_disk_space(path, required_bytes):
    try:
        usage = shutil.disk_usage(str(path.parent if path.parent.exists() else path.anchor))
        return usage.free > required_bytes * 1.1
    except Exception:
        return True


def download_file_resumable(url, dest, expected_md5=None, expected_size=None):
    if requests is None:
        return {"status": "error", "error": "requests library not installed"}
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = {"url": url, "dest": str(dest), "status": "pending"}

    if dest.exists() and expected_md5:
        existing_md5 = hashlib.md5(open(dest, "rb").read()).hexdigest()
        if existing_md5 == expected_md5:
            result["status"] = "already_complete"
            result["md5"] = existing_md5
            result["size"] = dest.stat().st_size
            return result

    if expected_size and not check_disk_space(dest, expected_size):
        result["status"] = "error"
        result["error"] = f"Insufficient disk space for {expected_size / 1e9:.2f} GB"
        return result

    existing_size = dest.stat().st_size if dest.exists() else 0
    headers = {}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"

    try:
        resp = requests.get(url, headers=headers, stream=True, timeout=60)
        if resp.status_code == 416:
            result["status"] = "already_complete"
            result["size"] = existing_size
            return result
        if resp.status_code not in (200, 206):
            result["status"] = "error"
            result["error"] = f"HTTP {resp.status_code}"
            return result

        total = int(resp.headers.get("content-length", 0)) + existing_size
        mode = "ab" if resp.status_code == 206 else "wb"
        print(f"  Downloading: {dest.name} ({total / 1e6:.1f} MB)")
        downloaded = existing_size
        t0 = time.time()

        with open(dest, mode) as f:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    elapsed = time.time() - t0
                    speed = (downloaded - existing_size) / max(1, elapsed) / 1e6
                    pct = 100.0 * downloaded / max(1, total) if total > 0 else 0
                    print(f"\r    [{pct:5.1f}%] {downloaded / 1e6:.1f}/{total / 1e6:.1f} MB ({speed:.1f} MB/s)",
                          end="", flush=True)
        print(f"\n  [OK] Downloaded {dest.name}")
        result["status"] = "downloaded"
        result["size"] = dest.stat().st_size

        if expected_md5:
            actual_md5 = hashlib.md5(open(dest, "rb").read()).hexdigest()
            result["md5"] = actual_md5
            if actual_md5 != expected_md5:
                result["status"] = "md5_mismatch"
                result["error"] = f"Expected {expected_md5}, got {actual_md5}"
        return result
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        return result


def extract_zip(archive, dest_dir):
    if not archive.exists():
        return False
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting: {archive.name}")
    try:
        with zipfile.ZipFile(str(archive), "r") as zf:
            zf.extractall(str(dest_dir))
        print(f"  [OK] Extracted")
        return True
    except Exception as e:
        print(f"  [ERROR] Extraction failed: {e}")
        return False


def download_dataset(dataset_key):
    info = DATASET_REGISTRY[dataset_key]
    result = {
        "dataset": dataset_key,
        "name": info["name"],
        "source_url": info["source_url"],
        "license": info["license"],
        "status": "pending",
        "files": [],
    }
    if info.get("blocker"):
        result["status"] = "blocked"
        result["blocker"] = info["blocker"]
        print(f"\n[BLOCKED] {info['name']}: {info['blocker']}")
        return result

    dest_dir = DATASETS_ROOT / dataset_key
    dest_dir.mkdir(parents=True, exist_ok=True)

    all_ok = True
    for file_info in info["files"]:
        file_dest = dest_dir / file_info["filename"]
        dl = download_file_resumable(
            file_info["url"], file_dest,
            expected_md5=file_info.get("expected_md5"),
            expected_size=file_info.get("size_bytes"),
        )
        result["files"].append(dl)
        if dl["status"] in ("downloaded", "already_complete"):
            if file_dest.suffix.lower() == ".zip":
                extract_dir = dest_dir / file_dest.stem
                if not extract_dir.exists() or not any(extract_dir.rglob("*")):
                    extract_zip(file_dest, dest_dir)
        else:
            all_ok = False
    result["status"] = "ready" if all_ok else "partial"
    return result


def write_inventory(results):
    INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INVENTORY_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset_key", "name", "source_url", "license",
            "status", "blocker", "file_count", "total_size_bytes",
            "task", "polarization", "notes"
        ])
        for r in results:
            info = DATASET_REGISTRY[r["dataset"]]
            total_size = sum(fi.get("size", 0) for fi in r.get("files", []))
            writer.writerow([
                r["dataset"], r["name"], r["source_url"], r["license"],
                r["status"], r.get("blocker", ""),
                len(r.get("files", [])), total_size,
                info["task"], info["polarization"], info["notes"]
            ])
    print(f"\n[INVENTORY] Written to {INVENTORY_PATH}")


def download_all_datasets():
    print("=" * 70)
    print("SpillSense External Validation - Dataset Download")
    print("=" * 70)
    results = []
    for key in DATASET_REGISTRY:
        print(f"\n{'=' * 60}")
        print(f"  Dataset: {DATASET_REGISTRY[key]['name']} [{key}]")
        print(f"{'=' * 60}")
        r = download_dataset(key)
        results.append(r)
    write_inventory(results)
    return results


if __name__ == "__main__":
    download_all_datasets()
