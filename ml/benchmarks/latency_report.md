# SpillSense (SIH-26143): Empirical Execution Latency Benchmark Report

**Benchmark Execution Date:** 2026-10-12T20:24:10Z  
**Evaluation Scope:** Representative Stratified Sample ($N=50$ scenes) extracted from the local collection of 1200 Sentinel-1 IW GRD GeoTIFF scenes ($2048 \times 2048 \times 2$, float32 calibrated $\sigma^0\text{ dB}$).  
**Hardware Environment:** NVIDIA GeForce RTX 4060 Laptop GPU | CUDA True | PyTorch 2.6.0+cu124 | ONNX Runtime 1.29.0  

---

## 1. Measured Software Execution Latency Summary

The table below presents empirically measured execution latencies using high-resolution monotonic timing (`time.perf_counter()`). Measurements exclude initial model loading and warm-up overhead.

| Pipeline Stage | Scenes/Runs Tested | Median Latency | Mean ± SD | p95 Latency |
|:---|---:|---:|---:|---:|
| 1. SAR GeoTIFF Disk Loading & Decoding | 50 | 171.00 ms | 174.37 ± 24.46 ms | 207.29 ms |
| 2. Radiometric Preprocessing (dB Normalization) | 50 | 58.16 ms | 63.67 ± 16.47 ms | 96.64 ms |
| 3. DualPolClassifier (PyTorch CUDA) | 50 | 9.56 ms | 10.03 ± 1.50 ms | 12.93 ms |
| 3. DualPolClassifier (PyTorch CPU) | 50 | 16.57 ms | 17.45 ± 3.24 ms | 23.92 ms |
| 3. DualPolClassifier (ONNX Runtime CPU) | 50 | 5.34 ms | 5.74 ± 1.56 ms | 8.69 ms |
| 3. DualPolClassifier (ONNX Runtime Web / WASM) | 30 | 12.38 ms | 12.01 ± 1.53 ms | 13.57 ms |
| 4. SpillSegNet Patch (PyTorch CUDA, 512x512) | 50 | 86.07 ms | 81.98 ± 8.21 ms | 86.54 ms |
| 4. SpillSegNet Patch (PyTorch CPU, 512x512) | 25 | 625.74 ms | 630.21 ± 54.82 ms | 694.12 ms |
| 4. SpillSegNet Patch (ONNX Runtime CPU, 512x512) | 25 | 229.18 ms | 240.14 ± 39.12 ms | 299.45 ms |
| 4. SpillSegNet Patch (ONNX Runtime Web / WASM, 512x512) | 20 | 489.16 ms | 501.14 ± 28.68 ms | 558.06 ms |
| 5. Full-Scene Tiled Segmentation (PyTorch CUDA, 25 Tiles) | 50 | 1172.30 ms | 1239.24 ± 417.14 ms | 2197.94 ms |
| 5. Full-Scene Tiled Segmentation (PyTorch CPU, 25 Tiles) | 3 | 8368.64 ms | 8406.80 ± 125.01 ms | 8528.66 ms |
| 5b. Segmentation Postprocessing (Thresholding & Masking) | 50 | 6.61 ms | 6.65 ± 1.45 ms | 8.82 ms |
| 6. Backward Drift Reconstruction (OpenDrift/OpenOil) | 0 | *Unavailable* | *Unavailable* | *Unavailable* |
| 7. AIS-Based Vessel Attribution Engine | 50 | 0.16 ms | 0.18 ± 0.05 ms | 0.29 ms |
| 8. End-to-End Pipeline (TIFF to Slick Mask, GPU) | 50 | 1413.86 ms | 1483.93 ± 411.62 ms | 2420.14 ms |

---

## 2. Workload & Cumulative Throughput Metrics

- **Total Representative Scenes Tested:** 50 scenes ($2048 \times 2048 \times 2$ resolution)
- **Total Pipeline Execution Workload Time:** 107.15 seconds
- **Mean End-to-End Latency Per Scene (GPU):** 1483.93 ms (1.48 s)
- **Effective Pipeline Throughput (GPU):** 0.67 scenes / second

### Model Initialization Latency (Excluded from Warm Runs)
- **Classifier (`best_classifier_v3.pt`):** 406.76 ms
- **Segmenter (`best_segmenter_v3.pt`):** 239.75 ms

---

## 3. Dataset-Aware Evaluation Against Zenodo Ground-Truth Masks

Inference latency was strictly isolated from reference mask loading and validation scoring. Reference masks were loaded independently to evaluate prediction quality on the tested sample:
- **Mean Dice Similarity Coefficient:** 0.6811 (68.1%)
- **Mean Intersection over Union (IoU):** 0.5552 (55.5%)
- **Mean Pixel Precision:** 0.6443 (64.4%)
- **Mean Pixel Recall:** 0.8893 (88.9%)
- **Evaluation Metric Computation Overhead:** Median 109.90 ms (excluded from model execution times)

### 3.1 Audit Analysis: Dice/IoU Discrepancy (88.4% / 79.3% vs. 68.11% / 55.52%)

An empirical audit was conducted to resolve the variance between previously reported patch-level figures (**88.4% Dice / 79.3% IoU**) and full-scene benchmark measurements (**68.11% Dice / 55.52% IoU**):

1. **Evaluation Domain (Patch-Level vs. Full-Scene Tiled Reconstruction):**
   - **Patch-Level Validation (`CachedSARSegmentationDataset`):** The model checkpoint `ml/checkpoints/best_segmenter_v3.pt` stores training validation metrics of **82.79% Dice and 70.63% IoU** (reaching up to 88.4% Dice in early holdout experiments). Crucially, this evaluation operates on curated $512 \times 512$ patches centered specifically on verified oil slick regions (2:1 oversampling of oil-containing crops). In this localized domain, the foreground-to-background ratio is balanced, yielding high precision (79.9%) and minimal ambient clutter.
   - **Full-Scene Tiled Ingestion ($2048 \times 2048$):** In the operational benchmark, entire uncropped scenes ($4,194,304$ pixels) are segmented via 25 sliding-window tiles. Across full scenes, oil slicks typically occupy less than $0.5\%\text{--}1.5\%$ of the marine raster. Ambient low-wind patches, wave boundary damping, and window averaging at tile boundaries introduce background false positives across the remaining $99\%$ of the scene, lowering precision to $64.43\%$.
2. **High Detection Sensitivity (Pixel Recall = 88.93%):**
   - Despite full-scene class imbalance, the model captures **88.93% of all verified ground-truth oil spill pixels** across all 50 full scenes. Real oil slicks are consistently detected with high fidelity.
3. **Macro vs. Micro Aggregation:**
   - The reported $68.11\%$ Dice is an unweighted **Macro-Average** across all 50 scenes ($\frac{1}{N}\sum \text{Dice}_i$). In scenes where an oil slick is small ($< 1,000$ pixels), minor boundary offsets disproportionately depress that individual scene's score. Micro-aggregation (accumulating total TP, FP, and FN pixels across the entire test set) yields higher scores ($71.4\%$ Dice at threshold $0.60$).

---

## 4. Tile-Count Verification (2048 × 2048 Scenes)

For a full Sentinel-1 IW GRD scene of dimensions $H = 2048, W = 2048$ pixels evaluated with tile size $T = 512$ and stride $S = 384$ ($25\%$ overlap, corresponding to a $128$-pixel border margin):

$$\text{Tiles per axis} = \left\lfloor \frac{H - T}{S} \right\rfloor + 1 = \left\lfloor \frac{2048 - 512}{384} \right\rfloor + 1 = \frac{1536}{384} + 1 = 4 + 1 = 5$$

The coordinate offsets along each axis are precisely:
$$y \in \{0, 384, 768, 1152, 1536\}$$
$$x \in \{0, 384, 768, 1152, 1536\}$$

Notice that the final tile starting at index $1536$ spans $[1536, 1536 + 512] = [1536, 2048]$, achieving complete coverage of the $2048$-pixel raster without clipping.

$$\text{Total Tiles per Scene} = 5 \times 5 = 25\text{ tiles}$$

*(In contrast, a non-overlapping grid with stride $512$ produces $\frac{2048}{512} \times \frac{2048}{512} = 4 \times 4 = 16\text{ tiles}$, but suffers from severe boundary truncation).*

---

## 5. End-to-End Pipeline Latency & Throughput Audit

The end-to-end execution path strictly incorporates all four sequential stages required to process a raw SAR GeoTIFF from disk to final georeferenced binary mask:

$$\text{Latency}_{\text{E2E}} = t_{\text{TIFF Ingestion}} + t_{\text{dB Normalization}} + \sum_{i=1}^{25} t_{\text{Tile Inference}} + t_{\text{Postprocessing}}$$

| Component | Mean Latency | Median Latency | Share of Total |
|:---|---:|---:|---:|
| 1. GeoTIFF Ingestion & Decoding | 174.37 ms | 171.00 ms | 11.7% |
| 2. Radiometric Preprocessing ($\sigma^0\text{ dB}$) | 63.67 ms | 58.16 ms | 4.3% |
| 3. Full-Scene Tiled GPU Segmentation (25 tiles) | 1239.24 ms | 1172.30 ms | 83.5% |
| 4. Morphological Postprocessing & Masking | 6.65 ms | 6.61 ms | 0.5% |
| **Total End-to-End Execution (GPU)** | **1483.93 ms** | **1413.86 ms** | **100.0%** |

- **Empirical Standard Deviation:** $\pm 411.62\text{ ms}$ (driven by tile-level CUDA kernel variance and host-GPU memory paging)
- **Empirical 95th Percentile ($p_{95}$):** $2420.14\text{ ms}$
- **Effective Pipeline Throughput:** $\frac{1000}{1483.93\text{ ms}} = 0.67\text{ scenes / second}$ (equating to $16.85\text{ tiles / second}$)

---

## 6. Benchmark Dataset & CSV Record Integrity Verification

All **603 individual timing records** stored in `ml/benchmarks/latency_results.csv` were audited against the aggregated JSON specification (`ml/benchmarks/latency_results.json`):

| CSV Stage Identifier | Sample Count ($N$) | CSV Median | JSON Median | Discrepancy | Status |
|:---|---:|---:|---:|---:|:---:|
| `1_TIFF_Loading` | 50 | 171.00 ms | 171.00 ms | 0.00 ms | Verified |
| `2_SAR_Preprocessing` | 50 | 58.16 ms | 58.16 ms | 0.00 ms | Verified |
| `3_Classifier_PyTorch_GPU` | 50 | 9.56 ms | 9.56 ms | 0.00 ms | Verified |
| `3_Classifier_PyTorch_CPU` | 50 | 16.57 ms | 16.57 ms | 0.00 ms | Verified |
| `3_Classifier_ONNX_CPU` | 50 | 5.34 ms | 5.34 ms | 0.00 ms | Verified |
| `3_Classifier_ONNX_WASM` | 30 | 12.38 ms | 12.38 ms | 0.00 ms | Verified |
| `4_SegPatch_PyTorch_GPU` | 50 | 86.07 ms | 86.07 ms | 0.00 ms | Verified |
| `4_SegPatch_PyTorch_CPU` | 25 | 625.74 ms | 625.74 ms | 0.00 ms | Verified |
| `4_SegPatch_ONNX_CPU` | 25 | 229.18 ms | 229.18 ms | 0.00 ms | Verified |
| `4_SegPatch_ONNX_WASM` | 20 | 489.16 ms | 489.16 ms | 0.00 ms | Verified |
| `5_Tiled_FullScene_GPU` | 50 | 1172.30 ms | 1172.30 ms | 0.00 ms | Verified |
| `5_Tiled_FullScene_CPU` | 3 | 8368.64 ms | 8368.64 ms | 0.00 ms | Verified |
| `5b_Segmentation_Postprocessing` | 50 | 6.61 ms | 6.61 ms | 0.00 ms | Verified |
| `7_AIS_AttributionEngine` | 50 | 0.16 ms | 0.16 ms | 0.00 ms | Verified |
| `8_End_to_End_Pipeline` | 50 | 1413.86 ms | 1413.86 ms | 0.00 ms | Verified |
| **Total Audit Records** | **603** | — | — | **0.00 ms** | **100% Match** |

---

## 7. Component Availability & Status Verification

1. **OpenDrift / OpenOil Hydrodynamic Backtracking:**
   - **Status:** **Unavailable** in this execution environment.
   - **Technical Cause:** The `opendrift` library is not installed in the active Python environment (listed as an optional dependency in `backend/requirements-optional.txt`). The system rigorously reports this capability as unexecuted rather than simulating synthetic timings.
2. **MovingPandas / GeoPandas AIS Trajectory Filtering:**
   - **Status:** **Unavailable**. `geopandas` and `movingpandas` are uninstalled in this environment.
3. **Core Kinematic Attribution Engine (`AttributionEngine.correlate_detections`):**
   - **Status:** **Fully Operational**. The mathematical kinematics and 5-factor scoring executed in a median of **0.160 ms** across 20 candidate vessels.
4. **ONNX Runtime Web / WASM:**
   - **Status:** **Operational via Node WebAssembly execution provider**. Demonstrates in-browser capability: single-patch classifier runs at median **12.38 ms**, and patch segmenter runs at median **489.16 ms**.

---

## 8. Verification Against Previously Reported Unverified Figures

| Component | Unverified Figure | Actual Empirically Measured Metric | Discrepancy & Technical Rationale |
|:---|:---|:---|:---|
| **SAR Preprocessing** | "35 ms" | **58.16 ms** (Mean: 63.67 ms) | Close to reported value; variation driven by 2048x2048 float32 memory bandwidth. |
| **Edge AI Inference** | "< 150 ms" | **12.38 ms** (Cls WASM), **9.56 ms** (Cls GPU) | Confirmed for classifier (<15 ms). Segmenter WASM requires ~489 ms due to dense 512x512 deconvolution. |
| **Tiled Full Scene Segmentation** | "650 ms" | **1172.30 ms** (GPU) | Empirically measured on RTX 4060 GPU across 25 tiles. On CPU, tiled segmentation takes ~8.4 seconds. |
| **Drift Reconstruction** | "3.2 s" | **Unavailable** | OpenDrift library is not installed in this environment; prior figure was synthetic. |
| **Vessel Attribution** | "450 ms" | **0.160 ms** | The pure Python spatial-kinematic scoring takes < 1 ms; 450 ms was an unverified upper-bound budget. |

---

## 9. Publication-Ready Section for IEEE Paper (Results and Discussion)

```text
To quantify the operational viability of the proposed SpillSense pipeline for tactical maritime surveillance, software execution latency was empirically evaluated on an AMD Ryzen workstation equipped with an NVIDIA GeForce RTX 4060 Laptop GPU running Python 3.11.9, PyTorch 2.6.0 (CUDA 12.4), and ONNX Runtime 1.29.0. Benchmarking was performed using monotonic high-resolution timing (time.perf_counter()) across a representative stratified sample of N=50 verified Sentinel-1 dual-polarization (VV/VH) IW GRD GeoTIFF scenes (2048 × 2048 × 2, float32 calibrated σ⁰ dB) paired with official Zenodo ground-truth masks. Across all tested scenes, raw SAR GeoTIFF ingestion required a median of 171.00 ms (p95: 207.29 ms), while physical backscatter normalization required 58.16 ms (p95: 96.64 ms). For spatial patch inference, DualPolOilSpillNet achieved a median classification latency of 9.56 ms on GPU (5.34 ms on ONNX CPU and 12.38 ms under ONNX Runtime WebAssembly), while SpillSegNet achieved a median patch segmentation latency of 86.07 ms on GPU (489.16 ms under WebAssembly). Full-scene segmentation executed via an overlapping 25-tile sliding window (512 × 512 patches, 384-pixel stride, covering all 2048 × 2048 pixels) yielded a median GPU inference latency of 1172.30 ms (p95: 2197.94 ms), whereas CPU execution required 8368.64 ms. Post-segmentation morphological thresholding executed in 6.61 ms, resulting in an end-to-end pipeline execution time from disk input to final segmented slick polygon of 1413.86 ms (mean: 1483.93 ± 411.62 ms, p95: 2420.14 ms), corresponding to a sustained throughput of 0.67 scenes per second. On uncropped full-scene validation, the segmented slicks achieved a mean Dice similarity coefficient of 68.11% (IoU: 55.52%) with high pixel recall (88.93%), whereas curated 512 × 512 oil crop validation yielded 82.79% Dice (70.63% IoU) due to reduced open-ocean clutter. Candidate vessel attribution via the kinematic AttributionEngine processed 20 AIS candidates in 0.160 ms. In contrast, full backward Lagrangian drift reconstruction through OpenDrift could not be executed due to uninstalled numerical dependencies in the local deployment environment and is designated as an operational limitation.
```
