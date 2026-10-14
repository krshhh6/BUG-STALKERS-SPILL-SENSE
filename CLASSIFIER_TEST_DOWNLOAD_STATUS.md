# SpillSense SIH-26143 — Classifier Independent Test Download & Staging Status

**Report Date:** 2026-10-14  
**Audit Author:** Antigravity Pair-Programming Assistant  
**Git Branch:** `improvement/phase0-backup` (HEAD: `d6e7730`)  
**Component:** DualPolOilSpillNet Classifier  
**Operational Status:** **CANDIDATE HELD IN STAGING — RESUMABLE TEST DOWNLOAD IN PROGRESS**

---

## 1. Current Deployment & Staging Posture

In strict adherence to the governance mandates in [`PROMOTION_DECISION.md`](file:///d:/Spill%20Sense/SIH-26143-OIL-Spill/PROMOTION_DECISION.md) and [`PHASE4_FINAL_VALIDATION.md`](file:///d:/Spill%20Sense/SIH-26143-OIL-Spill/PHASE4_FINAL_VALIDATION.md):

- **Active Production Classifier:** `best_dualpol_classifier.pt` / `frontend/public/models/oil_classifier.onnx` remains **100% active and untouched**.
- **Candidate Classifier:** `best_dualpol_classifier_exp2_sampling_sw30.pt` remains **frozen in staging** (SHA-256: `014D209A19DBD5C42DA275648F8E5BCB748ABCFCB1562B91F0541FBDCCE0653E`, operating threshold: **0.43**).
- **Validation Score Substitution Policy:** Validation metrics (0.9767 F1 / 96.24% recall) are **strictly withheld from serving as test scores**. Promotion is blocked until true out-of-distribution evaluation against Zenodo Part III is performed.

---

## 2. Official Zenodo Record & Archive Manifest

The independent test dataset has been verified on the official Zenodo repository:

| Metadata Field | Value |
|---|---|
| **Dataset Title** | Sentinel-1 SAR Oil spill image dataset for train, validate, and test deep learning models. Part III |
| **Zenodo Record ID** | `13761290` (Concept ID: `13761289`) |
| **Official DOI** | `10.5281/zenodo.13761290` |
| **Authors** | Trujillo-Acatitla et al. (2024) |
| **Remote Archive Filename** | `02_Test_images_and_ground_truth.7z` |
| **Remote Archive Size** | **9,859,650,011 bytes** (9.86 GB / 9.18 GiB) |
| **Expected MD5 Checksum** | `5dce64cd7ff9d80189d13504bd3bcbf5` |
| **Download Endpoint** | `https://zenodo.org/records/13761290/files/02_Test_images_and_ground_truth.7z?download=1` |
| **Intended Content** | 450 unseen SAR acquisitions: 150 Oil, 150 No-Oil, 150 Look-alike scenes ($2048 \times 2048$, dB) + 450 binary masks |

---

## 3. Resumable Background Downloader Configuration

Because the Zenodo server connection on this system operates at ~0.045 MB/s, an autonomous, resilient downloader worker has been deployed to execute in the background:

- **Worker Script:** [`ml/scripts/download_part3_resumable.py`](file:///d:/Spill%20Sense/SIH-26143-OIL-Spill/ml/scripts/download_part3_resumable.py)
- **Background Daemon Task ID:** `25db623a-508e-4afc-86e1-5f2457389b42/task-561`
- **Destination Folder:** `D:\Test`
- **Staging / Partial File:** `D:\Test\02_Test_images_and_ground_truth.7z.part`
- **Final Archive Target:** `D:\Test\02_Test_images_and_ground_truth.7z`
- **Persistent Human Log:** [`ml/logs/part3_download.log`](file:///d:/Spill%20Sense/SIH-26143-OIL-Spill/ml/logs/part3_download.log)
- **Persistent Machine Status:** [`ml/logs/part3_download_status.json`](file:///d:/Spill%20Sense/SIH-26143-OIL-Spill/ml/logs/part3_download_status.json)

### Safety & Resumption Features
1. **HTTP Range Headers:** If disconnected, the worker reads `existing_bytes` from `.part` and requests `bytes={existing_bytes}-`, appending chunks via `mode="ab"`.
2. **Atomic Promotion:** The `.part` extension ensures partial downloads cannot be mistaken for complete archives. The file is renamed to `.7z` only after completion.
3. **Exponential Backoff:** Network dropouts are retried automatically up to 1,000 times without failing.

---

## 4. Real-Time Download Status (Snapshot)

```json
{
  "phase": "DOWNLOADING",
  "bytes_downloaded": 3145728,
  "total_bytes": 9859650011,
  "percent": 0.03,
  "speed_mb_s": 0.040,
  "eta_hours": 68.49,
  "partial_file": "D:\\Test\\02_Test_images_and_ground_truth.7z.part",
  "attempt": 1,
  "last_updated_utc": "2026-10-14T15:38:55Z"
}
```

---

## 5. Post-Download Verification & Evaluation Protocol

Once the download stream terminates, the automated worker executes the following sequential pipeline:

```
[Archive Download Complete]
         │
         ▼
[Step 1: Checksum Verification]
   Compute MD5 on D:\Test\02_Test_images_and_ground_truth.7z
   Assert: Computed MD5 == 5dce64cd7ff9d80189d13504bd3bcbf5
         │
         ▼ (Pass)
[Step 2: Safe Extraction via py7zr]
   Extract to D:\Test\Extracted
         │
         ▼ (Pass)
[Step 3: Manifest Validation]
   Verify 450 TIFF scenes:
     - 150 Oil Spill Scenes + Masks
     - 150 No-Oil Clean Scenes + Masks
     - 150 Look-alike Formations + Masks
         │
         ▼ (Pass)
[Step 4: Independent Frozen Evaluation]
   Run evaluation on exactly the same test patches:
     - Baseline: best_dualpol_classifier.pt (Threshold: 0.49)
     - Candidate: best_dualpol_classifier_exp2_sampling_sw30.pt (Threshold: 0.43)
     - ZERO RETRAINING, ZERO THRESHOLD TUNING
         │
         ▼
[Step 5: Predefined Acceptance Gate]
   Check Promotion Criteria:
     ✓ Oil Recall ≥ 90.0%
     ✓ Ship-Wake FPR ≤ 5.0%
     ✓ Overall F1 ≥ 0.950
         │
   ┌─────┴────────────────┐
   ▼ (All Pass)           ▼ (Any Fail)
[PROMOTE TO PROD]      [RETAIN IN STAGING]
Export candidate ONNX   Keep baseline in prod
```

---

## 6. Commands to Monitor Download

To inspect download progress at any time:

```powershell
# View live JSON status
Get-Content "ml/logs/part3_download_status.json"

# Tail live download log
Get-Content "ml/logs/part3_download.log" -Tail 20
```

---

*Report generated and validated by SpillSense Phase 4 Release Suite.*  
*Artifact: `CLASSIFIER_TEST_DOWNLOAD_STATUS.md`*
