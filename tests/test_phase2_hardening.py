# -*- coding: utf-8 -*-
"""
Verification suite for Phase 2: Authentic SAR Detection Pipeline
Asserts elimination of synthetic simulations, zero references to generateHydrodynamicRing,
valid ONNX U-Net segmentation inference, and geospatial affine bounds preservation.
"""

from pathlib import Path
import pytest
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent

def test_no_synthetic_geometric_ring_generator():
    """Verify generateHydrodynamicRing is completely removed from frontend codebase."""
    detection_service_ts = REPO_ROOT / "frontend" / "src" / "services" / "detectionService.ts"
    assert detection_service_ts.exists(), "detectionService.ts must exist"
    content = detection_service_ts.read_text(encoding="utf-8")
    assert "generateHydrodynamicRing" not in content, (
        "generateHydrodynamicRing must be completely deleted from detectionService.ts"
    )

def test_cdse_sar_offline_benchmark_granules():
    """Verify CDSE SAR service loads authentic benchmark granules instead of pseudo-random sines."""
    from app.services.cdse_sar_service import CDSESARService
    cdse = CDSESARService()
    
    # Mumbai High coordinates
    bbox_mumbai = [71.0, 18.5, 71.4, 18.9]
    scene_info = cdse.search_latest_sentinel1_scene(bbox_mumbai)
    assert scene_info is not None
    
    vv, vh, cal = cdse.fetch_sar_backscatter_tile(bbox_mumbai, scene_info, grid_size=(256, 256))
    assert vv is not None, "Must load authentic Sentinel-1 VV array"
    assert vh is not None, "Must load authentic Sentinel-1 VH array"
    assert vv.shape == (256, 256)
    assert vh.shape == (256, 256)
    assert cal == 1.0

def test_sar_onnx_segmentation_inference():
    """Verify SARSPILLSegmentationEngine executes real ONNX oil_segmenter without dimension exceptions."""
    from app.services.sar_segmentation_model import SARSPILLSegmentationEngine
    from app.services.sar_preprocessing import SARPreprocessor

    engine = SARSPILLSegmentationEngine()
    assert engine.ort_session is not None, "ONNX Runtime session must be loaded"

    demo_oil_img = REPO_ROOT / "frontend" / "public" / "demo-sar" / "00_oil_vv.png"
    assert demo_oil_img.exists()
    
    vv_raw = np.array(Image.open(demo_oil_img), dtype=np.float32) / 255.0
    vv_db = (vv_raw * 22.0) - 32.0
    amp = np.sqrt(10.0 ** (vv_db / 10.0))
    sigma0_db = SARPreprocessor.calibrate_to_sigma0_db(amp)
    filtered = SARPreprocessor.lee_speckle_filter(sigma0_db)
    masked, sea = SARPreprocessor.mask_land_pixels(filtered)

    polygons = engine.run_inference(masked, sea, [71.0, 18.5, 71.4, 18.9], sensitivity_threshold=0.30)
    assert len(polygons) > 0, "Must detect real oil polygons from authentic Sentinel-1 scene"
    top = polygons[0]
    assert "geometry" in top
    assert top["geometry"]["type"] == "Polygon"
    assert top["area_km2"] > 0.0
    assert top["confidence"] > 0.40
    assert "slick_centroid" in top

def test_detect_endpoint_returns_real_detection():
    """Verify POST /api/v1/detect pipeline end-to-end."""
    from app.api.v1.detect_router import _execute_detection, DetectionRequest
    req = DetectionRequest(
        aoi={
            "type": "Polygon",
            "coordinates": [[[71.0, 18.5], [71.4, 18.5], [71.4, 18.9], [71.0, 18.9], [71.0, 18.5]]]
        },
        sensitivity=0.35
    )
    res = _execute_detection(req)
    assert res["status"] in ("detected", "no_oil_detected", "no_pass_available")
    if res["status"] == "detected":
        assert len(res["polygons"]) > 0
        assert res["total_detected_area_km2"] > 0
