# SpillSense (SIH-26143) — Unresolved Items & Blockers Catalog

**Audit Timestamp:** October 14, 2026  
**Auditor:** Team BUG STALKERS / Antigravity AI Engineering Pair  
**Branch:** `main` (commit `aad3a10`)

---

## 1. Overview
This document records all genuine technical and data blockers that cannot be solved without currently unavailable external data, execution environments, or missing ground-truth labels. No blocker has been concealed or replaced with synthetic estimates.

---

## 2. Catalog of Genuine Blockers

### Blocker 1: Zenodo Part III Archive Download in Progress
- **Item:** Independent test set archive `02_Test_images_and_ground_truth.7z` from Zenodo Record `13761290`.
- **Expected Archive Size:** 9,859,650,011 bytes (9.86 GB).
- **Expected MD5 Checksum:** `5dce64cd7ff9d80189d13504bd3bcbf5`.
- **Current Status:**
  - Active worker: `task-561` running `ml/scripts/download_part3_resumable.py`.
  - Progress: **1,298,137,088 bytes (1.30 GB / 13.17%)**.
  - Current Transfer Rate: ~0.057 MB/s.
  - Estimated Remaining Duration: **~41.5 hours**.
- **Impact on Release:**
  - The candidate classifier checkpoint (`best_dualpol_classifier_exp2_sampling_sw30.pt`) cannot be evaluated on the independent test set until the archive has completely downloaded and its MD5 verified.
  - Evaluation harness `ml/scripts/evaluate_part3_independent.py` is fully prepared and dry-run verified; it will execute immediately once the archive extracts.
- **Resolution Path:**
  - Allow `task-561` to run to completion undisturbed.
  - Run `python ml/scripts/evaluate_part3_independent.py` upon extraction.
  - Maintain candidate classifier in staging until execution finishes.

---

### Blocker 2: Absence of Ship-Wake Ground-Truth Annotations in Zenodo Part III
- **Item:** Promotion Gate 2 requirement ($\text{Ship-wake FPR} \le 5.0\%$) evaluated on independent data.
- **Root Cause Analysis:**
  - An inspection of Zenodo Record `13761290` (Part III test archive) confirms that it contains 450 scenes divided into three discrete classes:
    1. 150 Oil Spill scenes
    2. 150 Clean Ocean scenes
    3. 150 Look-alike scenes (algae, biogenic film, low wind)
  - **Zero ship-wake annotations, bounding boxes, or AIS vessel cross-references exist in Zenodo Part III.**
  - Ship wakes are not labelled as a distinct category.
- **Scientific Impact:**
  - Ship-wake FPR is **mathematically UNMEASURABLE** on Zenodo Part III alone.
  - Fabricating synthetic wake masks or substituting clean-ocean scenes for ship wakes would invalidate scientific integrity.
  - **Part III alone cannot satisfy Gate 2.**
- **Resolution Path:**
  - Construct a dedicated, independently published Sentinel-1 Ship-Wake Challenge Dataset by pairing Sentinel-1 IW GRD acquisitions with verified terrestrial/satellite AIS vessel tracks.
  - Evaluate both baseline ($\tau=0.49$) and candidate ($\tau=0.43$) on this separate wake dataset to measure independent ship-wake FPR.

---

### Blocker 3: OpenDrift / OpenOil Backward Drift Simulation Library Uninstalled
- **Item:** 4D Lagrangian backward drift reconstruction (Stage 6 of pipeline).
- **Current Status:**
  - The `opendrift` Python package and its dependencies (`netCDF4`, `cftime`, atmospheric forcing models) are **NOT INSTALLED** in the current Python 3.11 environment.
  - `latency_results.json` records Stage 6 as `count: 0, status: "Unavailable"`.
- **Scientific Impact:**
  - No backward drift simulation was executed during the automated benchmarking suite.
  - The IEEE paper must **NOT** claim quantitative runtime or empirical drift accuracy for OpenDrift.
- **Resolution Path:**
  - Clearly state OpenDrift/OpenOil integration as an architectural module and theoretical framework in the paper.
  - Present kinematic AIS attribution (`AttributionEngine`, 0.16 ms median latency) as the verified, executed attribution mechanism.

---

### Blocker 4: Real-Time ECMWF Copernicus CDS API Key Availability
- **Item:** Automated wind-field look-alike filtering via Copernicus Marine Service / ECMWF ERA5.
- **Current Status:**
  - The preprocessing pipeline includes wind-speed boundary checks, but dynamic ERA5 retrieval requires active ECMWF CDS API credentials not stored in the repository.
- **Scientific Impact:**
  - Wind speed look-alike filtering was validated using cached meteorological metadata, not dynamic real-time satellite downloads.
- **Resolution Path:**
  - Document wind-speed look-alike rejection as an algorithmic filter rather than an online API microservice.
