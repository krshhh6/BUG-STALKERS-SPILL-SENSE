"""
SpillSense (SIH-26143) — Zenodo Dataset Downloader
===================================================
Downloads and extracts the official Sentinel-1 SAR Oil Spill dataset
from Zenodo (Parts I, II, III).

Usage:
    python -m ml.download_zenodo [--part 1|2|3|all] [--dest D:/]
"""

import os
import sys
import time
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import requests

# Zenodo download URLs (direct download links)
ZENODO_FILES = {
    "part1": {
        "name": "Part I — Oil Spill Images & Masks",
        "files": [
            {
                "url": "https://zenodo.org/records/8346860/files/01_Train_Val_Oil_Spill_images.7z?download=1",
                "filename": "01_Train_Val_Oil_Spill_images.7z",
                "extract_to": "01_Train_Val_Oil_Spill_images",
                "description": "1200 Oil Spill SAR scenes (2048x2048x2, Sigma0 dB)"
            },
            {
                "url": "https://zenodo.org/records/8346860/files/01_Train_Val_Oil_Spill_mask.7z?download=1",
                "filename": "01_Train_Val_Oil_Spill_mask.7z",
                "extract_to": "01_Train_Val_Oil_Spill_mask",
                "description": "1200 Ground-truth binary masks (2048x2048, fg=1 bg=0)"
            },
        ]
    },
    "part2": {
        "name": "Part II — No-Oil & Look-alike Images & Masks",
        "files": [
            {
                "url": "https://zenodo.org/records/8253899/files/01_Train_Val_No_Oil_Images.7z?download=1",
                "filename": "01_Train_Val_No_Oil_Images.7z",
                "extract_to": "01_Train_Val_No_Oil_Images",
                "description": "685 Oil-free SAR scenes"
            },
            {
                "url": "https://zenodo.org/records/8253899/files/01_Train_Val_No_Oil_mask.7z?download=1",
                "filename": "01_Train_Val_No_Oil_mask.7z",
                "extract_to": "01_Train_Val_No_Oil_mask",
                "description": "685 No-oil ground-truth masks (all zeros)"
            },
            {
                "url": "https://zenodo.org/records/8253899/files/01_Train_Val_Lookalike_images.7z?download=1",
                "filename": "01_Train_Val_Lookalike_images.7z",
                "extract_to": "01_Train_Val_Lookalike_images",
                "description": "685 Look-alike SAR scenes"
            },
            {
                "url": "https://zenodo.org/records/8253899/files/01_Train_Val_Lookalike_mask.7z?download=1",
                "filename": "01_Train_Val_Lookalike_mask.7z",
                "extract_to": "01_Train_Val_Lookalike_mask",
                "description": "685 Look-alike ground-truth masks (all zeros)"
            },
        ]
    },
    "part3": {
        "name": "Part III — Test Set (Oil, No-Oil, Look-alike)",
        "files": [
            {
                "url": "https://zenodo.org/records/13761290/files/02_Test_images_and_ground_truth.7z?download=1",
                "filename": "02_Test_images_and_ground_truth.7z",
                "extract_to": "Test",
                "description": "450 Test SAR scenes and ground-truth masks (9.18 GB)"
            },
        ]
    },
}


def download_file(url: str, dest_path: Path, desc: str = "") -> bool:
    """Download a file with progress display. Supports resume."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Check for existing partial download
    existing_size = dest_path.stat().st_size if dest_path.exists() else 0

    headers = {}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)

        if response.status_code == 416:
            print(f"  [SKIP] {dest_path.name} already fully downloaded ({existing_size / 1e9:.2f} GB)")
            return True

        if response.status_code not in (200, 206):
            print(f"  [ERROR] HTTP {response.status_code} for {url}")
            return False

        total_size = int(response.headers.get("content-length", 0)) + existing_size
        mode = "ab" if response.status_code == 206 else "wb"

        print(f"  Downloading: {desc}")
        print(f"  File: {dest_path.name} ({total_size / 1e9:.2f} GB)")

        with open(dest_path, mode) as f:
            downloaded = existing_size
            t0 = time.time()
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):  # 8MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    elapsed = time.time() - t0
                    speed = (downloaded - existing_size) / max(1, elapsed) / 1e6
                    pct = 100.0 * downloaded / max(1, total_size)
                    print(f"\r  [{pct:5.1f}%] {downloaded / 1e9:.2f}/{total_size / 1e9:.2f} GB  ({speed:.1f} MB/s)", end="", flush=True)

        print(f"\n  [OK] Downloaded {dest_path.name}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"\n  [ERROR] Download failed: {e}")
        return False


def extract_7z(archive_path: Path, extract_to: Path) -> bool:
    """Extract a .7z archive using py7zr."""
    try:
        import py7zr
    except ImportError:
        print("  [ERROR] py7zr not installed. Run: pip install py7zr")
        return False

    if not archive_path.exists():
        print(f"  [ERROR] Archive not found: {archive_path}")
        return False

    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting: {archive_path.name} → {extract_to}")

    try:
        with py7zr.SevenZipFile(str(archive_path), mode="r") as z:
            z.extractall(path=str(extract_to))
        print(f"  [OK] Extracted to {extract_to}")
        return True
    except Exception as e:
        print(f"  [ERROR] Extraction failed: {e}")
        return False


def download_and_extract(parts: List[str], dest_root: Path = Path("D:/")):
    """Download and extract specified dataset parts."""
    print("=" * 60)
    print("SpillSense — Zenodo Dataset Downloader")
    print("=" * 60)
    print(f"Destination: {dest_root}\n")

    for part_key in parts:
        if part_key not in ZENODO_FILES:
            print(f"[WARN] Unknown part: {part_key}")
            continue

        part = ZENODO_FILES[part_key]
        print(f"\n{'─' * 50}")
        print(f"  {part['name']}")
        print(f"{'─' * 50}")

        for file_info in part["files"]:
            archive_path = dest_root / file_info["filename"]
            extract_dir = dest_root / file_info["extract_to"]

            # Check if already extracted
            if extract_dir.exists() and any(extract_dir.rglob("*.tif")):
                tif_count = sum(1 for _ in extract_dir.rglob("*.tif"))
                print(f"\n  [SKIP] {file_info['extract_to']} already extracted ({tif_count} TIFFs)")
                continue

            # Download
            print()
            if not download_file(file_info["url"], archive_path, file_info["description"]):
                continue

            # Extract
            if not extract_7z(archive_path, extract_dir):
                continue

    print(f"\n{'=' * 60}")
    print("Download complete. Run dataset status check:")
    print("  python -c \"from ml.config import CONFIG; CONFIG.data.print_status()\"")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Zenodo Oil Spill Dataset")
    parser.add_argument("--part", default="all", choices=["1", "2", "3", "all"],
                        help="Which part to download (1, 2, 3, or all)")
    parser.add_argument("--dest", default="D:/", type=str,
                        help="Destination root directory")
    args = parser.parse_args()

    if args.part == "all":
        parts = ["part1", "part2", "part3"]
    else:
        parts = [f"part{args.part}"]

    download_and_extract(parts, Path(args.dest))
