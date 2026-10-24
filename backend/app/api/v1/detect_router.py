# -*- coding: utf-8 -*-
"""
Spill Sense (SIH26143) — Real Sentinel-1 SAR + ML Segmentation Detection API
Endpoints:
- POST /api/v1/detect (and /api/detect)
- GET /api/v1/detect/status/{job_id}
"""

import uuid
import datetime
import os
from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks, File, UploadFile, Form
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional

from app.services.cdse_sar_service import CDSESARService
from app.services.sar_preprocessing import SARPreprocessor
from app.services.sar_segmentation_model import SARSPILLSegmentationEngine
from app.services.image_intelligence import (
    SARImageIntelligenceService,
    GeoTIFFReader,
    ImageIdentity,
    ImageModality
)
from app.services.tiled_inference import TiledSARInferenceEngine

router = APIRouter(tags=["Sentinel-1 SAR Detection Pipeline"])

# Singleton engine instances
segmentation_engine = SARSPILLSegmentationEngine()
tiled_engine = TiledSARInferenceEngine(segmentation_engine=segmentation_engine)


# Asynchronous job store
DETECTION_JOBS: Dict[str, Dict[str, Any]] = {}

class DetectionRequest(BaseModel):
    aoi: Optional[Dict[str, Any]] = Field(
        default=None,
        description="GeoJSON Polygon defining Area of Interest. If omitted, default incident AOI is used."
    )
    date_range: Optional[List[str]] = Field(
        default=None,
        description="[start_iso_date, end_iso_date] lookback range."
    )
    client_id: Optional[str] = Field(default=None, description="Optional CDSE Client ID")
    client_secret: Optional[str] = Field(default=None, description="Optional CDSE Client Secret")
    sensitivity: float = Field(default=0.35, ge=0.1, le=0.9, description="U-Net probability threshold")
    async_mode: bool = Field(default=False, description="Run detection asynchronously")

def _extract_bbox_from_aoi(aoi: Optional[Dict[str, Any]]) -> List[float]:
    """Extracts [min_lon, min_lat, max_lon, max_lat] from GeoJSON Polygon."""
    if not aoi or "coordinates" not in aoi:
        # Default to Mumbai High / Arabian Sea EEZ
        return [70.9, 18.4, 71.5, 19.0]

    coords = aoi["coordinates"]
    if not coords or not coords[0]:
        return [70.9, 18.4, 71.5, 19.0]

    ring = coords[0]
    lons = [pt[0] for pt in ring]
    lats = [pt[1] for pt in ring]
    return [min(lons), min(lats), max(lons), max(lats)]

def _execute_detection(req: DetectionRequest) -> Dict[str, Any]:
    bbox = _extract_bbox_from_aoi(req.aoi)
    cdse = CDSESARService(client_id=req.client_id, client_secret=req.client_secret)

    end_date = req.date_range[1] if (req.date_range and len(req.date_range) > 1) else None
    scene_info = cdse.search_latest_sentinel1_scene(bbox=bbox, lookback_days=30, end_date=end_date)

    if not scene_info:
        return {
            "status": "no_recent_imagery",
            "message": "No Sentinel-1 IW GRD passes found over the requested AOI within lookback window (30 days).",
            "aoi": req.aoi,
            "polygons": [],
            "model_version": segmentation_engine.MODEL_VERSION,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
        }

    # Pull calibrated SAR backscatter tile
    vv_arr, vh_arr, cal_factor = cdse.fetch_sar_backscatter_tile(bbox, scene_info, grid_size=(256, 256))

    # Preprocessing Pipeline:
    # 1. Radiometric calibration to sigma0 (dB)
    sigma0_db = SARPreprocessor.calibrate_to_sigma0_db(vv_arr, cal_factor)
    # 2. Adaptive Lee speckle filtering
    filtered_db = SARPreprocessor.lee_speckle_filter(sigma0_db, window_size=5)
    # 3. Authoritative Land-sea masking
    bbox_dict = {"min_lon": bbox[0], "min_lat": bbox[1], "max_lon": bbox[2], "max_lat": bbox[3]}
    masked_sar, sea_mask = SARPreprocessor.mask_land_pixels(filtered_db, bounds=bbox_dict)

    # ML Inference via PyTorch U-Net (SOS benchmark)
    polygons = segmentation_engine.run_inference(
        sar_sigma0_db=masked_sar,
        sea_mask=sea_mask,
        bbox=bbox,
        sensitivity_threshold=req.sensitivity,
        min_area_km2=0.05
    )

    if not polygons:
        return {
            "status": "no_oil_detected",
            "scene_timestamp": scene_info["scene_timestamp"],
            "sensor": scene_info["sensor"],
            "scene_id": scene_info["name"],
            "polarization": scene_info["polarization"],
            "orbit_direction": scene_info["orbit_direction"],
            "polygons": [],
            "total_detected_area_km2": 0.0,
            "model_version": segmentation_engine.MODEL_VERSION,
            "message": f"No hydrocarbon signature detected in the most recent Sentinel-1 pass ({scene_info['scene_timestamp'][:10]})."
        }

    total_area = round(sum(p["area_km2"] for p in polygons), 2)

    return {
        "status": "detected",
        "scene_timestamp": scene_info["scene_timestamp"],
        "sensor": scene_info["sensor"],
        "scene_id": scene_info["name"],
        "polarization": scene_info["polarization"],
        "orbit_direction": scene_info["orbit_direction"],
        "polygons": polygons,
        "total_detected_area_km2": total_area,
        "model_version": segmentation_engine.MODEL_VERSION,
        "message": f"Active hydrocarbon signature detected across {len(polygons)} slick segment(s) totaling {total_area} km²."
    }

@router.post("/detect")
@router.post("/v1/detect")
async def detect_oil_spill(req: DetectionRequest, background_tasks: BackgroundTasks):
    """
    Executes end-to-end Sentinel-1 SAR oil spill detection:
    1. Finds latest Sentinel-1 IW GRD pass over AOI in Copernicus Data Space Ecosystem.
    2. Performs radiometric calibration to sigma0 (dB) + Lee speckle filtering.
    3. Runs PyTorch U-Net segmentation trained on Sentinel-1 SAR Oil Spill (SOS) dataset.
    4. Converts raster mask to vector polygons with geodesic area (km²).
    """
    if req.async_mode:
        job_id = f"job-{uuid.uuid4().hex[:12]}"
        DETECTION_JOBS[job_id] = {
            "status": "PROCESSING",
            "job_id": job_id,
            "started_at": datetime.datetime.utcnow().isoformat() + "Z",
            "result": None
        }

        def run_async_task(jid: str, request_data: DetectionRequest):
            try:
                res = _execute_detection(request_data)
                DETECTION_JOBS[jid]["status"] = "COMPLETED"
                DETECTION_JOBS[jid]["result"] = res
            except Exception as ex:
                DETECTION_JOBS[jid]["status"] = "FAILED"
                DETECTION_JOBS[jid]["error"] = str(ex)

        background_tasks.add_task(run_async_task, job_id, req)
        return {
            "job_id": job_id,
            "status": "PROCESSING",
            "message": "Sentinel-1 pass query and SAR U-Net inference initiated."
        }

    return _execute_detection(req)

@router.get("/detect/status/{job_id}")
@router.get("/v1/detect/status/{job_id}")
async def get_detection_status(job_id: str):
    """Retrieves status and output of an asynchronous detection job."""
    if job_id not in DETECTION_JOBS:
        raise HTTPException(status_code=404, detail="Detection job ID not found.")
    return DETECTION_JOBS[job_id]


# ─────────────────────────────────────────────────────────────────────────────
# NEW PRODUCTION ENDPOINTS: SAR IMAGE INTELLIGENCE & TILED DETECTION
# ─────────────────────────────────────────────────────────────────────────────

class InspectPathRequest(BaseModel):
    file_path: str = Field(..., description="Absolute or relative path to satellite raster or image")


@router.post("/inspect-image")
@router.post("/v1/inspect-image")
@router.post("/detect/inspect")
@router.post("/v1/detect/inspect")
async def inspect_uploaded_image(
    file: Optional[UploadFile] = File(None),
    file_path: Optional[str] = Form(None)
):
    """
    Inspects an uploaded GeoTIFF, TIFF, or preview image before inference:
    - Extracts dimensions, band count, CRS, geotransform, bounding box, NoData.
    - Computes exact SHA-256 and normalized-pixel decoding invariance hash.
    - Evaluates radar quality indicators (speckle index, dynamic range, SNR).
    - Classifies modality (SAR Grayscale, Colorized SAR, Optical RGB, Multispectral, Overlay).
    - Generates high-contrast percentile-stretched web preview.
    """
    raw_bytes: bytes = b""
    filename: str = "uploaded_scene.tif"

    if file:
        raw_bytes = await file.read()
        filename = file.filename or filename
    elif file_path:
        p = Path(file_path)
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        raw_bytes = p.read_bytes()
        filename = p.name
    else:
        raise HTTPException(status_code=400, detail="Must provide either multipart 'file' or 'file_path'.")

    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="Provided file content is empty.")

    try:
        report = SARImageIntelligenceService.inspect(raw_bytes, filename=filename)
        return report
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Error inspecting raster: {str(ex)}")


@router.post("/detect-raster")
@router.post("/v1/detect-raster")
@router.post("/detect/raster")
@router.post("/v1/detect/raster")
async def detect_spill_from_raster(
    file: Optional[UploadFile] = File(None),
    file_path: Optional[str] = Form(None),
    sensitivity: float = Form(0.35),
    min_area_km2: float = Form(0.05)
):
    """
    End-to-end production SAR raster detection pipeline:
    1. Validates and inspects uploaded raster (TIFF, GeoTIFF, or preview).
    2. Enforces modality admission: blocks optical, annotated, or corrupt files from entering the SAR U-Net.
    3. Preserves geospatial metadata (CRS, transform, WGS84 bounds).
    4. Executes tiled overlapping inference with 2D Hann window blending to eliminate seam artifacts.
    5. Returns vector polygons with geodesic area (km²), damping depression, and uncertainty metrics.
    """
    raw_bytes: bytes = b""
    filename: str = "uploaded_scene.tif"

    if file:
        raw_bytes = await file.read()
        filename = file.filename or filename
    elif file_path:
        p = Path(file_path)
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        raw_bytes = p.read_bytes()
        filename = p.name
    else:
        raise HTTPException(status_code=400, detail="Must provide either multipart 'file' or 'file_path'.")

    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="Provided file content is empty.")

    # 1. Inspect image and classify modality
    try:
        inspection = SARImageIntelligenceService.inspect(raw_bytes, filename=filename)
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Raster inspection failed: {str(ex)}")

    classification = inspection["classification"]
    meta = inspection["metadata"]

    # 2. Enforce strict modality admission
    if classification.get("blocked_from_inference", False):
        return {
            "status": "blocked_unsupported_modality",
            "image_id": inspection["image_id"],
            "filename": filename,
            "category": classification["category"],
            "confidence": classification["confidence"],
            "evidence": classification["evidence"],
            "suitable_for_sar_model": False,
            "warnings": classification.get("warnings", []),
            "message": (
                f"Inference rejected: Image classified as {classification['category']}. "
                "The Sentinel-1 SAR oil spill model strictly requires microwave radar backscatter."
            ),
            "preview_data_url": inspection["preview"]["data_url"],
            "polygons": [],
            "total_detected_area_km2": 0.0
        }

    # 3. Read raster array and geospatial parameters
    try:
        array, _ = GeoTIFFReader.read_array(raw_bytes)
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to read raster array: {str(ex)}")

    # 4. Run tiled overlapping inference
    try:
        tiled_result = tiled_engine.run_tiled_inference(
            sar_raster=array,
            transform=meta.get("transform"),
            bounds=meta.get("bounds"),
            sensitivity_threshold=sensitivity,
            min_area_km2=min_area_km2
        )
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Tiled inference failed: {str(ex)}")

    # 5. Build unified production response
    return {
        "status": tiled_result["status"],
        "image_id": inspection["image_id"],
        "filename": filename,
        "file_sha256": inspection["identity"]["file_sha256"],
        "pixel_sha256": inspection["identity"]["pixel_sha256"],
        "image_type": {
            "category": classification["category"],
            "confidence": classification["confidence"],
            "suitable_for_sar_model": classification["suitable_for_sar_model"],
            "evidence": classification["evidence"],
            "uncertainty_status": classification["uncertainty_status"]
        },
        "raster_metadata": {
            "width": meta.get("width"),
            "height": meta.get("height"),
            "bands": meta.get("bands"),
            "is_georeferenced": meta.get("is_georeferenced"),
            "crs": meta.get("crs"),
            "bounds": meta.get("bounds"),
            "nodata_value": meta.get("nodata_value")
        },
        "preprocessing_applied": classification.get("preprocessing_required", []),
        "model_name": "SpillSense-UNet-SOS-Tiled",
        "model_version": tiled_result["model_version"],
        "polygons": tiled_result["polygons"],
        "total_detected_area_km2": tiled_result["total_detected_area_km2"],
        "ocean_mean_db": tiled_result["ocean_mean_db"],
        "uncertainty": tiled_result["uncertainty"],
        "preview_data_url": inspection["preview"]["data_url"],
        "message": tiled_result["message"],
        "warnings": classification.get("warnings", [])
    }


@router.post("/compare-identity")
@router.post("/v1/compare-identity")
@router.post("/detect/compare-identity")
@router.post("/v1/detect/compare-identity")
async def compare_image_identity(
    file1: Optional[UploadFile] = File(None),
    file2: Optional[UploadFile] = File(None),
    path1: Optional[str] = Form(None),
    path2: Optional[str] = Form(None)
):
    """
    Compares two satellite images to distinguish:
    1. Exact binary file duplicates (matching SHA-256).
    2. Same decoded image with different file container/compression (matching pixel SHA-256).
    3. Visually similar / resized images (dHash Hamming distance <= 5).
    4. Structurally related / cropped images (Hamming distance <= 12).
    5. Unrelated images.
    """
    def get_bytes_and_name(f: Optional[UploadFile], p: Optional[str], default_name: str) -> Tuple[bytes, str]:
        if f:
            return f.file.read(), f.filename or default_name
        if p:
            pth = Path(p)
            if not pth.exists():
                raise HTTPException(status_code=404, detail=f"File not found: {p}")
            return pth.read_bytes(), pth.name
        raise HTTPException(status_code=400, detail=f"Must provide image file or path for {default_name}.")

    bytes1, name1 = get_bytes_and_name(file1, path1, "image1")
    bytes2, name2 = get_bytes_and_name(file2, path2, "image2")

    arr1, _ = GeoTIFFReader.read_array(bytes1)
    arr2, _ = GeoTIFFReader.read_array(bytes2)

    id1 = ImageIdentity.from_bytes_and_array(bytes1, arr1)
    id2 = ImageIdentity.from_bytes_and_array(bytes2, arr2)

    comparison = id1.compare(id2)

    return {
        "image_1": {"filename": name1, "identity": id1.to_dict()},
        "image_2": {"filename": name2, "identity": id2.to_dict()},
        "comparison": comparison
    }

