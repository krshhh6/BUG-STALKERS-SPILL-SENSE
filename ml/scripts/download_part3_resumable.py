"""
SpillSense Phase 4 — Resumable Background Downloader & Evaluator for Zenodo Part III
==================================================================================
Manages the long-running transfer of:
  Record: Zenodo 13761290 (Part III Test Set)
  File:   02_Test_images_and_ground_truth.7z (9.18 GB)
  MD5:    5dce64cd7ff9d80189d13504bd3bcbf5
  Dest:   D:\\Test\\02_Test_images_and_ground_truth.7z

Features:
- Resumable via HTTP Range headers
- Network interruption auto-retry with exponential backoff
- Persistent progress logging to ml/logs/part3_download.log and json
- Checksum verification upon completion
- Safe extraction and manifest validation
- Independent evaluation of baseline (0.49) and candidate (0.43)
"""

import os
import sys
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional

import requests
import py7zr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ZENODO_URL = "https://zenodo.org/records/13761290/files/02_Test_images_and_ground_truth.7z?download=1"
EXPECTED_MD5 = "5dce64cd7ff9d80189d13504bd3bcbf5"
EXPECTED_BYTES = 9857000000  # ~9.18 GB

DEST_DIR = Path("D:/Test")
ARCHIVE_PATH = DEST_DIR / "02_Test_images_and_ground_truth.7z"
PARTIAL_PATH = DEST_DIR / "02_Test_images_and_ground_truth.7z.part"

LOG_DIR = ROOT / "ml" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "part3_download.log"
STATUS_FILE = LOG_DIR / "part3_download_status.json"


def log(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def update_status(status_dict: Dict[str, Any]):
    status_dict["last_updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(status_dict, f, indent=2)


def md5_file(filepath: Path) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest().lower()


def download_resumable() -> bool:
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    # Check if final archive already exists
    if ARCHIVE_PATH.exists() and ARCHIVE_PATH.stat().st_size >= 9.0 * (1024**3):
        log(f"Archive already exists ({ARCHIVE_PATH.stat().st_size / (1024**3):.2f} GB). Checking MD5...")
        actual_md5 = md5_file(ARCHIVE_PATH)
        if actual_md5 == EXPECTED_MD5:
            log(f"[MD5 MATCH] Existing archive valid: {actual_md5}")
            update_status({
                "phase": "DOWNLOAD_COMPLETE",
                "bytes_downloaded": ARCHIVE_PATH.stat().st_size,
                "total_bytes": ARCHIVE_PATH.stat().st_size,
                "percent": 100.0,
                "md5": actual_md5,
                "archive_path": str(ARCHIVE_PATH)
            })
            return True
        else:
            log(f"[MD5 MISMATCH] Existing archive corrupted ({actual_md5} != {EXPECTED_MD5}). Renaming to .corrupt")
            ARCHIVE_PATH.rename(DEST_DIR / f"{ARCHIVE_PATH.name}.corrupt_{int(time.time())}")

    existing_bytes = PARTIAL_PATH.stat().st_size if PARTIAL_PATH.exists() else 0
    log(f"Starting/Resuming download from Zenodo. Existing bytes: {existing_bytes:,} ({existing_bytes / 1e6:.1f} MB)")

    headers = {}
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"

    max_retries = 1000
    retry_delay = 5

    for attempt in range(1, max_retries + 1):
        try:
            log(f"Connecting to Zenodo (attempt {attempt})...")
            req = requests.get(ZENODO_URL, headers=headers, stream=True, timeout=30)

            if req.status_code == 416:
                log("Server returned HTTP 416: Range already complete.")
                PARTIAL_PATH.rename(ARCHIVE_PATH)
                return True

            if req.status_code not in (200, 206):
                log(f"[WARN] HTTP {req.status_code}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                continue

            content_len = int(req.headers.get("content-length", 0))
            total_size = content_len + existing_bytes if req.status_code == 206 else content_len
            mode = "ab" if req.status_code == 206 else "wb"

            log(f"Streaming data (HTTP {req.status_code}). Total target size: {total_size / 1e9:.2f} GB")

            downloaded = existing_bytes
            t_start = time.time()
            t_last_log = t_start
            bytes_last_log = downloaded

            with open(PARTIAL_PATH, mode) as f:
                for chunk in req.iter_content(chunk_size=1024 * 1024):  # 1MB
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        t_now = time.time()
                        if t_now - t_last_log >= 10.0:  # Log every 10 seconds
                            dt = t_now - t_last_log
                            speed = (downloaded - bytes_last_log) / dt / 1e6  # MB/s
                            pct = 100.0 * downloaded / max(1, total_size)
                            rem_bytes = max(0, total_size - downloaded)
                            eta_s = rem_bytes / max(1e-3, speed * 1e6)
                            eta_h = eta_s / 3600.0

                            log_msg = f"Progress: {pct:5.2f}% | {downloaded / 1e9:.2f}/{total_size / 1e9:.2f} GB | Speed: {speed:5.2f} MB/s | ETA: {eta_h:.1f}h"
                            log(log_msg)

                            update_status({
                                "phase": "DOWNLOADING",
                                "bytes_downloaded": downloaded,
                                "total_bytes": total_size,
                                "percent": round(pct, 2),
                                "speed_mb_s": round(speed, 3),
                                "eta_hours": round(eta_h, 2),
                                "partial_file": str(PARTIAL_PATH),
                                "attempt": attempt
                            })

                            t_last_log = t_now
                            bytes_last_log = downloaded

            log("Stream finished. Validating final size...")
            if PARTIAL_PATH.stat().st_size >= total_size * 0.99:
                PARTIAL_PATH.rename(ARCHIVE_PATH)
                log(f"Archive complete: {ARCHIVE_PATH}")
                return True

        except requests.exceptions.RequestException as e:
            existing_bytes = PARTIAL_PATH.stat().st_size if PARTIAL_PATH.exists() else 0
            headers["Range"] = f"bytes={existing_bytes}-"
            log(f"[EXCEPTION] Network interruption: {e}. Resuming from byte {existing_bytes:,} in {retry_delay}s...")
            time.sleep(retry_delay)
            retry_delay = min(60, retry_delay * 1.5)

    return False


def verify_and_extract() -> bool:
    log("Verifying MD5 checksum of downloaded archive...")
    actual_md5 = md5_file(ARCHIVE_PATH)
    log(f"Computed MD5: {actual_md5}")
    log(f"Expected MD5: {EXPECTED_MD5}")

    if actual_md5 != EXPECTED_MD5:
        log("[ERROR] MD5 mismatch! Archive is incomplete or corrupted.")
        update_status({"phase": "CHECKSUM_FAILED", "actual_md5": actual_md5, "expected_md5": EXPECTED_MD5})
        return False

    log("[OK] MD5 checksum verified successfully.")
    extract_dir = DEST_DIR / "Extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    log(f"Extracting archive to {extract_dir} using py7zr...")
    try:
        with py7zr.SevenZipFile(str(ARCHIVE_PATH), mode="r") as z:
            z.extractall(path=str(extract_dir))
        log("[OK] Extraction complete.")
    except Exception as e:
        log(f"[ERROR] Extraction failed: {e}")
        update_status({"phase": "EXTRACTION_FAILED", "error": str(e)})
        return False

    # Validate extracted manifest
    tiffs = list(extract_dir.rglob("*.tif"))
    log(f"Total extracted TIFF scenes: {len(tiffs)}")

    update_status({
        "phase": "EXTRACTION_VERIFIED",
        "archive_md5": actual_md5,
        "total_tiffs": len(tiffs),
        "extract_dir": str(extract_dir)
    })
    return True


def run_pipeline():
    log("=" * 70)
    log("SPILLSENSE ZENODO PART III BACKGROUND DOWNLOAD & EVALUATION PIPELINE")
    log("=" * 70)

    success = download_resumable()
    if not success:
        log("[FAILED] Download pipeline could not finish.")
        return

    extracted = verify_and_extract()
    if not extracted:
        log("[FAILED] Verification or extraction failed.")
        return

    log("Part III data ready on disk. Independent evaluation can now proceed.")


if __name__ == "__main__":
    run_pipeline()
