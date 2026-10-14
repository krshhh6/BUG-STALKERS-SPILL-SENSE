# SpillSense (SIH-26143) — Dedicated Sentinel-1 Ship-Wake Challenge Dataset Specification

**Document Purpose:** Reproducible dataset specification and validation protocol to satisfy Promotion Gate 2 ($\text{Ship-wake FPR} \le 5.0\%$).  
**Originating Constraint:** Zenodo Record `13761290` (Part III test set) contains 450 scenes partitioned strictly as 150 Oil, 150 Clean, and 150 Look-alike, with **zero ship-wake annotations or AIS track cross-references**.

---

## 1. Problem Definition & Operational Hazard
In single-look complex and ground-range detected Sentinel-1 SAR imagery, ship turbulent wakes create linear surface-roughness damping patterns (Bragg wave suppression) that mimic the dark radar backscatter depression of biogenic and crude oil slicks. Without explicit wake annotations, false alarm rates from shipping traffic cannot be measured independently.

---

## 2. Satellite Acquisition & Scene Selection Criteria

### 2.1 Sensor & Mode Specification
- **Sensor:** Sentinel-1A / Sentinel-1B C-band SAR.
- **Acquisition Mode:** Interferometric Wide (IW) Swath, Ground Range Detected High Resolution (GRDH).
- **Polarimetric Channels:** Dual-polarization VV + VH (Level-1 GRD).
- **Pixel Spacing:** $10\text{ m} \times 10\text{ m}$ ($2048 \times 2048$ pixel scene frames).
- **Geographic Scope:** High-density maritime corridors (e.g., English Channel, Strait of Malacca, Singapore Strait, Mumbai High offshore sector).

### 2.2 Environmental Constraints
- **Surface Wind Speeds:** $3\text{ m/s} \le u_{10} \le 12\text{ m/s}$ (ECMWF ERA5 reanalysis verified) to ensure radar Bragg backscatter is active and wake boundaries remain distinct.
- **Incidence Angles:** $30^\circ \le \theta \le 45^\circ$ across the swath to avoid near-range saturation and far-range signal-to-noise deterioration.

---

## 3. Dataset Composition & Class Breakdown

The challenge set shall comprise **300 curated full-scene acquisitions** ($2048 \times 2048$) partitioned into:

| Subset Class | Scene Count | Ground Truth Verification Source | Role in Evaluation |
|---|---|---|---|
| **Class A: Verified Ship Wakes** | 100 scenes | Terrestrial / Satellite AIS trajectory match within $\pm 15\text{ minutes}$ of SAR pass | Primary wake false positive rate challenge |
| **Class B: Clean Ocean (Controls)** | 100 scenes | Zero vessel traffic, zero slicks, homogeneous sea clutter | Negative control for background speckle |
| **Class C: Natural Look-alikes** | 50 scenes | Low-wind pools ($u_{10} < 3\text{ m/s}$), algal blooms, internal waves | Negative control for natural ocean phenomena |
| **Class D: Confirmed Oil Slicks** | 50 scenes | Emergency maritime response agency confirmed spills | Sensitivity control ensuring oil recall remains $\ge 90\%$ |

---

## 4. Annotation & Adjudication Protocol

1. **AIS Track Correlation:**
   - AIS messages (MMSI, SOG, COG, Heading, Length, Width) interpolated along the SAR azimuth-range coordinates.
   - Doppler centroid velocity correction applied to account for SAR azimuth displacement of moving vessels.
2. **Double-Blind Expert Annotation:**
   - Two independent remote-sensing SAR analysts delineate wake features:
     - Kelvin wake arms (transverse and diverging waves)
     - Narrow-V turbulent wake centerline
     - Internal wave wake packets
3. **Adjudication Standard:**
   - Polygon agreement threshold: $\text{Intersection-over-Union} \ge 0.75$.
   - Any wake feature lacking AIS confirmation within 30 nautical miles is discarded from Class A.

---

## 5. Scene-Level Train / Validation / Test Separation

To guarantee zero data leakage:
- **Spatial Separation:** Minimum 150 km separation between training scenes and evaluation scenes.
- **Temporal Separation:** Acquisitions partitioned across distinct orbital repeat cycles (minimum 12-day offset).
- **Vessel Separation:** No vessel MMSI appearing in the training split may appear in the evaluation split.

---

## 6. Evaluation Metrics & Acceptance Criteria (Promotion Gate 2)

### 6.1 Mathematical Formulation
$$\text{Ship-Wake False Positive Rate (FPR)} = \frac{\text{False Positives on Wake Pixels/Patches}}{\text{Total Ground-Truth Wake Pixels/Patches}} = 1.0 - \text{Wake Rejection TNR}$$

### 6.2 Acceptance Thresholds
- **Mandatory Gate 2 Criterion:**
  $$\text{Ship-Wake FPR} \le 0.050 \quad (5.0\%)$$
- **Oil Recall Preservation (on Class D):**
  $$\text{Oil Detection Recall} \ge 0.900 \quad (90.0\%)$$
- **Overall F1 Score:**
  $$F_1 \ge 0.950$$

---

## 7. Current Governance Verdict
Until this dedicated dataset is downloaded, annotated, and evaluated:
- **Gate 2 Status:** **BLOCKED (UNMEASURABLE ON ZENODO PART III)**
- **Candidate Classifier Action:** **HOLD IN STAGING**
- **Production Classifier:** Maintain active baseline `best_dualpol_classifier.pt`.
