# 🤖 AI Agent & LLM Context Guide — SpillSense (SIH 26143)

> **FOR ALL AI AGENTS & LANGUAGE MODELS (Gemini, Claude, GPT, Cursor, Copilot, Aider, etc.)**
> 
> **STOP — DO NOT READ THE ENTIRE CODEBASE.**
> This file is the single source of truth for repository structure, architecture, runtime conventions, deployment rules, and critical implementation details. Reading this file gives you 100% of the operational context needed to modify, debug, or extend this project without wasting context window or execution tokens.

---

## 1. Project Overview & Mission

- **Project Name:** SpillSense — Maritime Intelligence C2 Platform
- **Competition / Track:** Smart India Hackathon (SIH 2026), Problem Statement **SIH 26143**
- **Domain:** Maritime Domain Awareness (MDA), Disaster Management, Environmental Intelligence
- **Core Mission:**
  1. **Spaceborne Radar AI Detection:** Automated Sentinel-1 SAR / ISRO EOS-04 detection of marine oil spills via dual-polarization (VV/VH) Bragg damping analysis.
  2. **In-Browser Edge AI Segmentation:** Client-side inference via ONNX Runtime Web (`ort-wasm-simd-threaded.jsep.wasm`) to detect oil slicks directly in browser with zero server load.
  3. **4D Lagrangian Hydrodynamic Drift:** Backward backtracking to find spill origin coordinates and forward forecasting to predict coastal impact using CMEMS/HYCOM ocean currents and wind vectors.
  4. **Dark Vessel AIS Forensic Attribution:** Correlates backward drift trajectories against live AIS (AISStream WebSocket) and historical vessel tracks (AISHub / DuckDB) to identify polluter vessels even if they turned off AIS ("dark vessels").
  5. **Cryptographic Court Dossiers:** Tamper-proof, SHA-256 sealed PDF evidence lockers admissible under UNCLOS and Section 356 of the Indian Merchant Shipping Act (1958).

---

## 2. Monorepo Architecture & Directory Map

```
D:\SIH 26143\
├── AGENTS.md                  # <-- YOU ARE HERE: Universal AI Agent manual
├── README.md                  # Human-facing project overview & SIH documentation
├── vercel.json                # Root Vercel build & route rewrite configuration
├── package.json               # Root npm scripts (proxy commands into frontend)
├── .env.example               # Root template for environment credentials
│
├── frontend/                  # React 19 + TypeScript + Vite SPA
│   ├── vercel.json            # Sub-directory Vercel config (matches root config)
│   ├── package.json           # Frontend dependencies (React 19, Cesium, ONNX Web, Leaflet)
│   ├── vite.config.ts         # Vite bundler, Cesium plugin, and ONNX wasm middleware
│   ├── public/
│   │   ├── cesium/            # CesiumJS static assets (Assets, Workers, ThirdParty, Widgets)
│   │   ├── onnx-dist/         # Pre-bundled ONNX Runtime Web .wasm and .mjs files
│   │   └── fonts/             # Google Material Symbols font assets
│   └── src/
│       ├── App.tsx            # Main application shell, view routing, scenario switcher
│       ├── types/dashboard.ts # Core TypeScript interfaces (Scenario, Vessel, Polygon, etc.)
│       ├── services/
│       │   ├── sarClassifier.ts         # Edge ONNX & Bragg damping inference engine
│       │   ├── detectionService.ts      # API bridge to backend SAR detection router
│       │   ├── marineWeatherService.ts  # Weather & ocean current vector fetching
│       │   └── socket.ts                # Real-time Socket.IO client connection
│       ├── components/
│       │   ├── views/
│       │   │   ├── GlobeView.tsx        # 3D Tactical Cesium globe with live AIS & slick pins
│       │   │   ├── DashboardView.tsx    # C2 overview metrics, vessel risk table & quick stats
│       │   │   ├── DetectionView.tsx    # SAR scene viewer, polygon masks & confidence score
│       │   │   ├── DriftView.tsx        # 4D backward/forward hydrodynamic drift simulation
│       │   │   ├── AttributionView.tsx  # Multi-factor AIS vessel scoring & anomaly radar
│       │   │   ├── EvidenceView.tsx     # Chain-of-custody cryptographic dossier generator
│       │   │   └── AnalyticsView.tsx    # Trend analysis, regional risk index, EEZ charts
│       │   ├── modals/
│       │   │   └── ApiKeySetupModal.tsx # Glassmorphism API modal (CDSE, AISStream, AISHub, etc.)
│       │   ├── LeafletMap.tsx           # 2D GIS ECDIS hydrographic Leaflet map engine
│       │   ├── MapPanel.tsx             # 2D/3D map viewport container with HUD overlay
│       │   ├── Topbar.tsx               # Maritime C2 header, live UTC clock, basin selector
│       │   └── Sidebar.tsx              # Tactical navigation sidebar
│
├── backend/                   # FastAPI Python C2 Backend
│   ├── requirements.txt       # Python dependencies (FastAPI, Uvicorn, DuckDB, NumPy, etc.)
│   ├── app/
│   │   ├── main.py            # FastAPI entry point, CORS, Socket.IO mount, startup daemon
│   │   ├── core/
│   │   │   ├── config.py      # App settings & environment variable resolution
│   │   │   └── socket_server.py # Real-time Socket.IO server for telemetry broadcasts
│   │   ├── api/v1/
│   │   │   ├── ais_router.py          # Live AIS vessels, spatial bounding queries, stats
│   │   │   ├── detect_router.py       # SAR scene detection & Bragg damping processing
│   │   │   ├── endpoints.py           # Historical incidents, scenarios, drift triggering
│   │   │   ├── setup_router.py        # API key verification & status checker
│   │   │   └── active_learning_router.py # Coast guard human-in-the-loop analyst feedback
│   │   ├── services/
│   │   │   ├── aisstream_service.py   # Continuous WebSocket listener to aisstream.io
│   │   │   ├── ais_history_service.py # In-memory & DuckDB trajectory ring buffer
│   │   │   ├── aishub_service.py      # REST fallback polling for AISHub vessel data
│   │   │   ├── cdse_sar_service.py    # Copernicus Data Space Ecosystem API client
│   │   │   ├── sar_segmentation_model.py # PyTorch SAR segmentation inference
│   │   │   └── pdf_evidence_generator.py # ReportLab PDF cryptographic court dossier builder
│   │   └── tasks/
│   │       ├── drift_simulation.py    # Lagrangian 4D Runge-Kutta numerical drift solver
│   │       └── ais_correlation.py     # Trajectory spatiotemporal intersection engine
│
├── ml/                        # Machine Learning, SAR Models & Training
│   ├── models/                # PyTorch architectures (U-Net, DANN, DualPolNet, ASPP)
│   ├── checkpoints/           # Trained model weights (.pt, .pth, .onnx)
│   ├── scripts/
│   │   ├── export_onnx.py     # Script to export PyTorch weights to browser ONNX models
│   │   └── train_segmentation.py # PyTorch training pipeline on Sentinel-1 SAR scenes
│   └── reports/               # Independent validation and false-positive audit reports
│
└── pipeline/                  # SAR Preprocessing & Ingestion
    ├── preprocess.py          # Radiometric calibration, speckle filter, dB scaling
    └── universal_loader.py    # GeoTIFF, SAFE format, and NumPy raster loaders
```

---

## 3. Tech Stack & Version Matrix

| Layer | Technology | Key Version | Role |
| :--- | :--- | :--- | :--- |
| **Frontend Framework** | React | `^19.2.8` | Component UI with StrictMode & concurrent features |
| **Frontend Language** | TypeScript | `~6.0.2` | Static typing throughout entire frontend |
| **Bundler / Build** | Vite | `^8.2.2` | Lightning build tool with Rollup/Rolldown chunking |
| **3D Geospatial Globe** | CesiumJS | `^1.146.0` | 3D orbital globe, high-altitude tactical rendering |
| **2D Hydrographic Map** | Leaflet | `^1.9.4` | ECDIS 2D navigational charts & nautical vector tiles |
| **In-Browser ML** | ONNX Runtime Web | `^1.29.0` | Edge wasm execution of SAR segmentation |
| **Styling** | Tailwind CSS | `^3.4.19` | Dark glassmorphism maritime C2 theme |
| **Backend Framework** | FastAPI | `^0.110.0+` | Asynchronous REST and WebSocket API |
| **Backend Language** | Python | `3.11` / `3.13` | ASGI microservice engine |
| **Realtime Push** | python-socketio | `^5.11.0` | Low-latency telemetry push to frontend |
| **AIS Data Stream** | websockets | `^12.0` | Persistent daemon connecting to AISStream |
| **Geospatial Storage** | DuckDB + PostGIS | `latest` | High-speed spatiotemporal trajectory queries |

---

## 4. Key Developer Commands

### Frontend Commands (Run from root or `frontend/`)
```powershell
# Install frontend dependencies (ALWAYS use --legacy-peer-deps for React 19 compatibility)
npm --prefix frontend install --legacy-peer-deps

# Start Vite dev server on http://localhost:5173
npm --prefix frontend run dev

# Run TypeScript type check
npm --prefix frontend run test

# Production build (Outputs to frontend/dist/ in ~11 seconds)
npm --prefix frontend run build
```

### Backend Commands (Run from `backend/`)
```powershell
# Install Python dependencies
pip install -r backend/requirements.txt

# Run FastAPI backend on http://127.0.0.1:8000
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### One-Click Launch (Runs both Backend + Frontend in parallel)
```powershell
.\start_localhost.ps1
```

---

## 5. Deployment Rules & Constraints

### ⚠️ Critical Architecture Rule: Vercel vs Persistent Backend
- **Frontend on Vercel:** **YES.** Vercel hosts the React Vite SPA, Cesium WebGL, and ONNX wasm files flawlessly across global edge CDNs.
- **Backend on Vercel:** **NO.** Vercel Serverless Functions have a 10–60 second maximum execution timeout and **do NOT support persistent WebSockets**.
- **Backend Production Hosts:** Deploy `backend/` to **Render** (`render.yaml` is pre-configured), **Railway**, **Fly.io**, or an **AWS / DigitalOcean VPS**.
- **Frontend API Endpoint Configuration:**
  - In local development, Vite proxies `/api` and `/socket.io` to `http://127.0.0.1:8000`.
  - In production on Vercel, set the environment variable:
    ```env
    VITE_API_URL=https://spillsense-api.onrender.com
    ```

### Vercel Configuration Details
- `vercel.json` exists in **both** root and `frontend/` to ensure builds pass regardless of whether the Vercel Root Directory is set to `./` or `frontend/`.
- Crucial header rules in `vercel.json`:
  - `.wasm` files MUST be served with `Content-Type: application/wasm` and `Cache-Control: immutable`.
  - `.mjs` files MUST be served with `Content-Type: application/javascript`.
  - Rewrites route all non-API paths to `/index.html` for client-side routing.

---

## 6. Codebase Quirks & Anti-Patterns (Must Avoid)

### A. CesiumJS 3D Globe Quirks ([GlobeView.tsx](file:///d:/SIH%2026143/frontend/src/components/views/GlobeView.tsx))
1. **Never use negative `eyeOffset`:** Setting `eyeOffset = new Cartesian3(0, 0, -25)` pushes labels and billboards 25 meters *under the earth ellipsoid surface*, making them invisible. Always keep `eyeOffset: Cartesian3.ZERO`.
2. **Always disable depth test on tactical billboards & points:**
   ```typescript
   disableDepthTestDistance: Number.POSITIVE_INFINITY
   ```
   Without this, billboards near sea level clip against the terrain mesh and disappear.
3. **Distance Display Conditions:** To make spill cards and incident markers visible from orbital altitudes (e.g. 3,800 km), set `distanceDisplayCondition: new Cesium.DistanceDisplayCondition(0, 15_000_000)`.
4. **Tile Boundary & Maximum Zoom:**
   - Clamped `viewer.scene.screenSpaceCameraController.minimumZoomDistance = 1500` to prevent zooming through the ocean surface.
   - ArcGIS World Imagery tile provider capped at `maximumLevel = 17` to prevent requesting non-existent level 18/19 tiles (which triggers "Map data not yet available" placeholders).
5. **High-DPI Billboards:** Do not use inline raw SVG strings directly in Cesium Billboard URLs (they render blurry on Retina/High-DPI displays). Use the HTML5 Canvas 2D texture generators (`getVesselPinBillboard`, `getGlassmorphicBadgeCard`).

### B. Backend Routing & API Conventions ([backend/app/main.py](file:///d:/SIH%2026143/backend/app/main.py))
1. **Dual Route Mounting:** Routers are mounted under **both** `/api/v1` and `/api` (e.g. `/api/v1/detect` and `/api/detect`). Always maintain this dual mount when adding new routers so that both direct backend calls and proxy rewrites work seamlessly.
2. **AISStream WebSocket Singleton:** `AISStreamService` in `backend/app/services/aisstream_service.py` is a thread-safe singleton. It launches a background task during FastAPI startup (`@app.on_event("startup")`) to stream live maritime positions into `AISTrajectoryBuffer`.

### C. ONNX Runtime Web Edge Inference ([sarClassifier.ts](file:///d:/SIH%2026143/frontend/src/services/sarClassifier.ts))
1. The ONNX wasm binary (`ort-wasm-simd-threaded.jsep.wasm`) is 28 MB.
2. In Vite, `onnxWasmPlugin` in `vite.config.ts` intercepts `/onnx-dist/` in dev and copies it to `dist/onnx-dist/` on production build.
3. If ONNX inference fails to load in the browser, check CORS headers and ensure `application/wasm` MIME type is sent.

### D. UI & Styling Rules
1. **Glassmorphism C2 Theme:**
   - Dark background: `#070b14` / `#0a0f1d` / `#0f172a`
   - Panels: `background: rgba(15, 23, 42, 0.75); backdrop-filter: blur(12px); border: 1px solid rgba(56, 189, 248, 0.15);`
   - Accent colors: Cyan `#06b6d4`, Blue `#38bdf8`, Emerald `#10b981`, Amber `#f59e0b`, Rose `#f43f5e`
2. **Anti-Slop Standard:**
   - No generic AI-generated placeholder buttons, dummy lorem ipsum, or non-functional controls.
   - Keep comments concise and architectural; never remove existing docstrings or explanatory comments.

---

## 7. Fast Navigation: Where to Find Core Features

| Feature | Primary Frontend File | Primary Backend File |
| :--- | :--- | :--- |
| **3D Tactical Globe & AIS Pins** | `frontend/src/components/views/GlobeView.tsx` | `backend/app/services/aisstream_service.py` |
| **2D ECDIS Nautical Chart** | `frontend/src/components/LeafletMap.tsx` | `backend/app/api/v1/endpoints.py` |
| **C2 Dashboard & Incidents** | `frontend/src/components/views/DashboardView.tsx` | `backend/app/api/v1/endpoints.py` |
| **SAR Detection & Bragg Damping** | `frontend/src/components/views/DetectionView.tsx` | `backend/app/api/v1/detect_router.py` |
| **Edge ONNX Model Inference** | `frontend/src/services/sarClassifier.ts` | `ml/scripts/export_onnx.py` |
| **4D Lagrangian Drift Simulation** | `frontend/src/components/views/DriftView.tsx` | `backend/app/tasks/drift_simulation.py` |
| **Vessel AIS Attribution** | `frontend/src/components/views/AttributionView.tsx` | `backend/app/services/attribution_engine.py` |
| **Court Dossier PDF Generator** | `frontend/src/components/views/EvidenceView.tsx` | `backend/app/services/pdf_evidence_generator.py` |
| **Copernicus CDSE API Client** | `frontend/src/services/detectionService.ts` | `backend/app/services/cdse_sar_service.py` |
| **API Key Setup Modal** | `frontend/src/components/modals/ApiKeySetupModal.tsx` | `backend/app/api/v1/setup_router.py` |
| **Types & Data Contracts** | `frontend/src/types/dashboard.ts` | `backend/app/models/` |

---

## 8. Checklist for AI Agents Before Submitting Work

1. [ ] **TypeScript Check:** Run `npm --prefix frontend run test` (or `npx tsc -b`) to verify no compilation errors.
2. [ ] **Production Build Check:** Run `npm --prefix frontend run build` to verify the Vite bundle generates cleanly.
3. [ ] **No Accidental Commit Pastes:** Ensure no bash/git outputs or commit strings leaked into source files.
4. [ ] **MIME & Headers Check:** If modifying wasm or assets, ensure `vercel.json` headers match.
5. [ ] **Secrets Safety:** Never hardcode secrets in source code; use `.env` and `app.core.config.settings`.
