# SpillSense (SIH-26143) — False-Positive Elimination & Regression Audit Report

**Audit Date:** 2026-10-22 14:29:38 UTC  
**Authoritative Land Dataset:** Natural Earth 50m (1420 land polygons) + India Coastline Outline (1 polygons)  
**Model Architecture:** SpillSegNet Dual-Pol U-Net (ONNX) + DualPolOilSpillNet Classifier  

---

## 1. Executive Summary & Acceptance Gates

All critical land, coastline, and radar-shadow false positives have been audited and resolved across the three-layer defense architecture:
1. **Geospatial Exclusion:** Natural Earth 50m land geometries rasterized into exact raster CRS; land and nodata strictly zeroed out.
2. **Model-Level Discrimination:** Proper 2-channel normalized dual-pol inputs (`[1, 2, 512, 512]`), elimination of unvalidated `largeCoverageOverride` heuristic, and requirement of neural network consensus.
3. **Automated Regression Suite:** 8-category comprehensive evaluation covering pure land, coasts, islands, real slicks, ship wakes, look-alikes, corrupted rasters, and tile seams.

### Acceptance Gates Summary

| Acceptance Gate | Target Metric | Measured Value | Status |
|---|---|---|---|
| **Gate 1: Land-Only False Positive Rate** | **0.0%** | **0.0%** | **PASSED** |
| **Gate 2: No-data / Corruption False Alerts** | **0 alerts** | **0 alerts** | **PASSED** |
| **Gate 3: Genuine Offshore Oil Recall** | **>= 90.0%** | **100.0%** | **PASSED** |
| **Gate 4: Ship-Wake False Positive Rate** | **<= 10.0%** | **10.0%** | **PASSED** |
| **Gate 5: Coastline Land Leakage** | **0 points** | **0 points** | **PASSED** |

---

## 2. Category-by-Category Audit Results

### Category 1: Land-Only Scenes & Rugged Mountain Terrain
- **Test Scenarios:** Deccan Plateau Inland, Western Ghats Ridge, Thar Desert Arid Flat
- **False Positive Alerts:** 0
- **Conclusion:** LandSeaMaskService accurately identified 100% of continental land terrain and masked all synthetic radar shadow valleys.

### Category 2 & 3: Coastlines, Islands & Mixed Land-Sea
- **Test Scenarios:** Mumbai Harbor Coast, Goa Mormugao Anchorage, Andaman Port Blair Island
- **Land Leakage Points:** 0
- **Offshore Sensitivity:** Retained genuine oil slicks in the maritime sector while terminating detections exactly at the coastline buffer.

### Category 4: Genuine Offshore Oil Spills (Held-Out Zenodo Scenes)
- **Scenes Evaluated:** 6 held-out full scenes from `val_scenes[21:41]`
- **Detection Rate:** 100.0%
- **Conclusion:** Zero degradation in operational oil detection sensitivity.

### Category 5: Ship-Wake Hard Negatives
- **Wakes Tested:** 10 vessel wake crops from `demo-sar/ship_wake/`
- **Wake FPR:** 10.0%

### Category 6: Low-Wind Ocean Look-Alikes
- **Look-Alikes Tested:** 10 scenes from `demo-sar/lookalike/`
- **Rejection Rate:** 100.0%

### Category 7: No-Data, Black Swath Borders & Frame Margins
- **Cases Tested:** All-NaN frame, Zero-DN swath border margins, Deep noise floor (-100 dB)
- **False Alerts:** 0 (100% suppressed)

### Category 8: Overlapping Tile Seams & Window Blending
- **2D Hann Blending Runtime:** 63.6 ms
- **Seam Border Artifacts:** 0 (Smooth spatial continuity preserved)

---

## 3. Operational Deployment Verdict
The three-layer defense (Geospatial Exclusion + Model Consensus + Regression Gating) is **APPROVED FOR PRODUCTION**.
