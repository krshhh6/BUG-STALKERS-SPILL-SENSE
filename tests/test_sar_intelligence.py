# -*- coding: utf-8 -*-
"""
Spill Sense (SIH-26143) — Automated SAR Image Intelligence Test Suite
=====================================================================
Unit and integration tests covering:
1. GeoTIFF reading, tag parsing, affine transform, and WGS84 bounding box calculation.
2. Single-band and dual-polarization (VV/VH) radar backscatter ingestion.
3. Image modality classification (SAR Grayscale, Colorized SAR, Optical RGB, Overlay, Unknown).
4. Multi-level image identity tracking (SHA-256, normalized pixel hash, perceptual dHash/aHash).
5. Percentile-based web preview generation without mutating original scientific rasters.
6. Overlapping tiled inference with 2D Hann window blending and geodesic polygon vectorization.
7. Full FastAPI API endpoint validation, modality blocking, and backward compatibility.
"""

import io
import os
import sys
import math
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

# Ensure backend is on sys.path
root_dir = Path(__file__).resolve().parent.parent
backend_dir = root_dir / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import tifffile
from fastapi.testclient import TestClient

from app.main import app
from app.services.image_intelligence import (
    ImageModality,
    IdentityMatchStatus,
    ImageIdentity,
    GeoTIFFReader,
    RasterQualityAnalyzer,
    ImageModalityClassifier,
    WebPreviewGenerator,
    SARImageIntelligenceService
)
from app.services.tiled_inference import TiledSARInferenceEngine
from app.services.sar_preprocessing import SARPreprocessor


class TestSARImageIntelligence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.temp_dir = tempfile.mkdtemp(prefix="spillsense_test_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def tearDown(self):
        import gc
        gc.collect()

    # ─────────────────────────────────────────────────────────────────────────
    # 1. GeoTIFF Reading and Metadata Extraction
    # ─────────────────────────────────────────────────────────────────────────
    def test_geotiff_reading_and_metadata(self):
        """Tests GeoTIFF reading, affine transform calculation, and WGS84 bounding box."""
        h, w = 128, 128
        arr = np.random.uniform(-30.0, -10.0, size=(h, w)).astype(np.float32)

        # GeoTIFF Tags:
        # Tag 33550 (ModelPixelScaleTag): [dx, dy, dz] = 0.001 deg (~110m)
        # Tag 33922 (ModelTiepointTag): [i, j, k, lon, lat, z] = [0, 0, 0, 71.0, 19.0, 0]
        # Tag 34735 (GeoKeyDirectoryTag): Geographic CRS EPSG 4326
        tags = [
            (33550, 'd', 3, (0.001, 0.001, 0.0), False),
            (33922, 'd', 6, (0.0, 0.0, 0.0, 71.0, 19.0, 0.0), False),
            (34735, 'H', 8, (1, 1, 0, 1, 2048, 0, 1, 4326), False)
        ]
        tiff_path = Path(self.temp_dir) / "test_geo.tif"
        tifffile.imwrite(str(tiff_path), arr, extratags=tags)

        meta = GeoTIFFReader.read_metadata(tiff_path, filename=tiff_path.name)
        self.assertEqual(meta["width"], 128)
        self.assertEqual(meta["height"], 128)
        self.assertEqual(meta["bands"], 1)
        self.assertTrue(meta["is_georeferenced"])
        self.assertIsNotNone(meta["bounds"])

        bounds = meta["bounds"]
        self.assertAlmostEqual(bounds["min_lon"], 71.0, places=3)
        self.assertAlmostEqual(bounds["max_lat"], 19.0, places=3)
        self.assertAlmostEqual(bounds["max_lon"], 71.0 + (128 * 0.001), places=3)
        self.assertAlmostEqual(bounds["min_lat"], 19.0 - (128 * 0.001), places=3)

        # Verify read_array
        loaded_arr, loaded_meta = GeoTIFFReader.read_array(tiff_path)
        self.assertEqual(loaded_arr.shape, (128, 128))
        self.assertEqual(loaded_arr.dtype, np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Dual-Polarization SAR (VV / VH) Ingestion
    # ─────────────────────────────────────────────────────────────────────────
    def test_dual_polarization_sar_ingestion(self):
        """Tests dual-channel (VV and VH) Sentinel-1 SAR scene ingestion."""
        h, w = 100, 100
        # Band 0 = VH (median ≈ -32 dB), Band 1 = VV (median ≈ -19 dB)
        vh = np.random.normal(-32.0, 2.0, size=(h, w)).astype(np.float32)
        vv = np.random.normal(-19.0, 2.5, size=(h, w)).astype(np.float32)
        dual_pol = np.stack([vh, vv], axis=0)  # Shape: (2, 100, 100)

        tiff_path = Path(self.temp_dir) / "dualpol_sar.tif"
        tifffile.imwrite(str(tiff_path), dual_pol)

        inspection = SARImageIntelligenceService.inspect(tiff_path)
        self.assertEqual(inspection["metadata"]["bands"], 2)
        cls_res = inspection["classification"]
        self.assertEqual(cls_res["category"], ImageModality.SAR_GRAYSCALE)
        self.assertTrue(cls_res["suitable_for_sar_model"])
        self.assertFalse(cls_res["blocked_from_inference"])
        self.assertGreaterEqual(cls_res["confidence"], 0.90)

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Image Modality Classification
    # ─────────────────────────────────────────────────────────────────────────
    def test_colorized_sar_detection(self):
        """Tests that colorized/pseudocolor SAR is recognized and flagged rather than mistaken for optical."""
        h, w = 120, 120
        # Synthetic 1D colormap (Jet-like: Blue at low values, Green in middle, Red at high values)
        scalar_radar = np.linspace(-35.0, -5.0, h * w).reshape((h, w)).astype(np.float32)
        norm = (scalar_radar - (-35.0)) / (30.0)

        # 1D colormapped curve in RGB
        r = np.clip(1.5 - np.abs(4.0 * norm - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * norm - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * norm - 1.0), 0.0, 1.0)
        colorized = (np.stack([r, g, b], axis=0) * 255.0).astype(np.uint8)

        img_path = Path(self.temp_dir) / "colorized_sar.png"
        Image.fromarray(np.transpose(colorized, (1, 2, 0))).save(img_path)

        inspection = SARImageIntelligenceService.inspect(img_path)
        cls_res = inspection["classification"]
        self.assertEqual(cls_res["category"], ImageModality.SAR_COLORIZED)
        self.assertTrue(cls_res["blocked_from_inference"])
        self.assertIn("warnings", cls_res)
        self.assertTrue(len(cls_res["warnings"]) > 0)

    def test_optical_rgb_imagery_blocking(self):
        """Tests that true optical RGB imagery is detected and blocked from SAR inference."""
        h, w = 120, 120
        # True ocean optical characteristics: deep water heavily absorbs red, blue dominates,
        # with green coastal sediments and independent clouds
        b = np.random.uniform(140, 200, size=(h, w)).astype(np.float32)
        g = np.random.uniform(60, 110, size=(h, w)).astype(np.float32)
        r = np.random.uniform(10, 45, size=(h, w)).astype(np.float32)

        optical_img = np.stack([r, g, b], axis=2).astype(np.uint8)
        img_path = Path(self.temp_dir) / "optical_satellite.png"
        Image.fromarray(optical_img).save(img_path)

        inspection = SARImageIntelligenceService.inspect(img_path)
        cls_res = inspection["classification"]
        self.assertEqual(cls_res["category"], ImageModality.OPTICAL_RGB)
        self.assertFalse(cls_res["suitable_for_sar_model"])
        self.assertTrue(cls_res["blocked_from_inference"])

    def test_annotated_overlay_detection(self):
        """Tests that images containing saturated overlay graphics or text borders are flagged."""
        h, w = 150, 150
        # Base grayscale radar
        base = np.random.uniform(50, 120, size=(h, w, 3)).astype(np.uint8)
        # Burn in bright saturated red bounding box and yellow text overlay
        base[10:20, 10:100] = [255, 0, 0]    # Pure red banner
        base[50:110, 50:55] = [255, 255, 0]  # Pure yellow line
        base[50:110, 100:105] = [255, 255, 0]

        img_path = Path(self.temp_dir) / "annotated_scene.png"
        Image.fromarray(base).save(img_path)

        inspection = SARImageIntelligenceService.inspect(img_path)
        cls_res = inspection["classification"]
        self.assertEqual(cls_res["category"], ImageModality.ANNOTATED_OVERLAY)
        self.assertTrue(cls_res["blocked_from_inference"])
        self.assertTrue(cls_res["evidence"]["has_overlay"])

    def test_invalid_and_corrupt_files(self):
        """Tests that 0-byte or corrupted files are rejected gracefully without unhandled exceptions."""
        empty_path = Path(self.temp_dir) / "empty_file.tif"
        empty_path.write_bytes(b"")

        # FastAPI inspect endpoint test
        with open(empty_path, "rb") as f:
            res = self.client.post("/api/v1/inspect-image", files={"file": ("empty.tif", f, "image/tiff")})
        self.assertEqual(res.status_code, 400)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Image Identity and Duplicate Detection
    # ─────────────────────────────────────────────────────────────────────────
    def test_exact_and_near_duplicate_identity(self):
        """Tests exact file duplicate, format re-encoding duplicate, and visually similar detection."""
        h, w = 100, 100
        arr = np.random.uniform(20, 200, size=(h, w)).astype(np.uint8)

        # File 1: PNG format
        p1 = Path(self.temp_dir) / "img_base.png"
        Image.fromarray(arr).save(p1)

        # File 2: Exact duplicate copy
        p2 = Path(self.temp_dir) / "img_copy.png"
        p2.write_bytes(p1.read_bytes())

        # File 3: Same decoded image saved as TIFF (different container, same pixels)
        p3 = Path(self.temp_dir) / "img_as_tiff.tif"
        tifffile.imwrite(str(p3), arr)

        # File 4: Visually similar resized image (90x90)
        p4 = Path(self.temp_dir) / "img_resized.png"
        Image.fromarray(arr).resize((90, 90)).save(p4)

        # File 5: Completely unrelated random image
        p5 = Path(self.temp_dir) / "img_unrelated.png"
        Image.fromarray(np.random.uniform(0, 255, size=(h, w)).astype(np.uint8)).save(p5)

        arr1, _ = GeoTIFFReader.read_array(p1)
        arr2, _ = GeoTIFFReader.read_array(p2)
        arr3, _ = GeoTIFFReader.read_array(p3)
        arr4, _ = GeoTIFFReader.read_array(p4)
        arr5, _ = GeoTIFFReader.read_array(p5)

        id1 = ImageIdentity.from_bytes_and_array(p1.read_bytes(), arr1)
        id2 = ImageIdentity.from_bytes_and_array(p2.read_bytes(), arr2)
        id3 = ImageIdentity.from_bytes_and_array(p3.read_bytes(), arr3)
        id4 = ImageIdentity.from_bytes_and_array(p4.read_bytes(), arr4)
        id5 = ImageIdentity.from_bytes_and_array(p5.read_bytes(), arr5)

        # 1. Exact Duplicate
        cmp_exact = id1.compare(id2)
        self.assertEqual(cmp_exact["status"], IdentityMatchStatus.EXACT_FILE_DUPLICATE)
        self.assertEqual(cmp_exact["confidence"], 1.0)

        # 2. Same Decoded Image (different file container / metadata)
        cmp_decoded = id1.compare(id3)
        self.assertEqual(cmp_decoded["status"], IdentityMatchStatus.SAME_DECODED_IMAGE)
        self.assertEqual(cmp_decoded["confidence"], 0.99)

        # 3. Visually Similar (resized version)
        cmp_similar = id1.compare(id4)
        self.assertIn(cmp_similar["status"], (IdentityMatchStatus.VISUALLY_SIMILAR, IdentityMatchStatus.STRUCTURALLY_RELATED))
        self.assertLessEqual(cmp_similar["hamming_distance"], 6)

        # 4. Unrelated Image
        cmp_unrelated = id1.compare(id5)
        self.assertEqual(cmp_unrelated["status"], IdentityMatchStatus.UNRELATED)
        self.assertGreater(cmp_unrelated["hamming_distance"], 10)

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Web Preview Generation
    # ─────────────────────────────────────────────────────────────────────────
    def test_web_preview_generation(self):
        """Tests percentile-based contrast stretching for web PNG previews."""
        arr = np.array([[-35.0, -25.0], [-15.0, -5.0]], dtype=np.float32)
        preview_bytes, b64_url = WebPreviewGenerator.generate_sar_preview(arr, p_low=0.0, p_high=100.0)

        self.assertTrue(len(preview_bytes) > 0)
        self.assertTrue(b64_url.startswith("data:image/png;base64,"))

        # Ensure original array was not modified
        self.assertEqual(arr[0, 0], -35.0)

        # Verify decoded preview is valid PNG
        dec = Image.open(io.BytesIO(preview_bytes))
        self.assertEqual(dec.format, "PNG")

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Tiled Overlapping Inference & Mask Alignment
    # ─────────────────────────────────────────────────────────────────────────
    def test_tiled_overlapping_inference(self):
        """Tests tiled sliding-window inference with 2D Hann window blending across large scene."""
        h, w = 600, 600
        # Ambient ocean clutter ≈ -15 dB
        scene_db = np.random.normal(-15.0, 1.5, size=(h, w)).astype(np.float32)

        # Introduce circular oil slick anomaly with strong capillary damping (-28 dB)
        yy, xx = np.ogrid[:h, :w]
        slick_mask = ((xx - 300) ** 2 + (yy - 300) ** 2) <= (60 ** 2)
        scene_db[slick_mask] = np.random.normal(-27.0, 1.0, size=int(np.sum(slick_mask))).astype(np.float32)

        tiled = TiledSARInferenceEngine(tile_size=400, overlap_fraction=0.30)
        result = tiled.run_tiled_inference(
            sar_raster=scene_db,
            transform=[0.001, 0.0, 71.0, 0.0, -0.001, 19.0],
            sensitivity_threshold=0.15,
            min_area_km2=0.01
        )

        self.assertIn("status", result)
        self.assertIn("polygons", result)
        self.assertIn("uncertainty", result)
        self.assertIn("ocean_mean_db", result)

        # Must detect the synthetic slick
        self.assertEqual(result["status"], "detected")
        self.assertGreater(len(result["polygons"]), 0)

        poly = result["polygons"][0]
        self.assertGreater(poly["area_km2"], 0.0)
        self.assertLess(poly["mean_damping_db"], -6.0)  # Significant negative damping
        self.assertIn("slick_centroid", poly)

        # Verify uncertainty metrics
        unc = result["uncertainty"]
        self.assertIn("mean_shannon_entropy", unc)
        self.assertIn("false_positive_risk", unc)

    # ─────────────────────────────────────────────────────────────────────────
    # 7. FastAPI API Endpoints & Backward Compatibility
    # ─────────────────────────────────────────────────────────────────────────
    def test_api_endpoints_and_backward_compatibility(self):
        """Tests FastAPI REST endpoints: health, static incidents, inspect-image, detect-raster, compare-identity."""
        # Health check
        res_health = self.client.get("/api/v1/health")
        self.assertEqual(res_health.status_code, 200)

        # Static incidents fallback
        res_static = self.client.get("/api/v1/incidents/static")
        self.assertEqual(res_static.status_code, 200)
        self.assertTrue(len(res_static.json()) >= 8)

        # Inspect GeoTIFF endpoint
        h, w = 80, 80
        arr = np.random.uniform(-30.0, -10.0, size=(h, w)).astype(np.float32)
        bio = io.BytesIO()
        tifffile.imwrite(bio, arr)
        bio.seek(0)

        res_inspect = self.client.post(
            "/api/v1/inspect-image",
            files={"file": ("test_scene.tif", bio.getvalue(), "image/tiff")}
        )
        self.assertEqual(res_inspect.status_code, 200)
        data = res_inspect.json()
        self.assertIn("image_id", data)
        self.assertIn("identity", data)
        self.assertIn("classification", data)
        self.assertIn("preview", data)
        self.assertEqual(data["classification"]["category"], "sar_grayscale")

        # Detect Raster endpoint with SAR GeoTIFF
        bio.seek(0)
        res_detect = self.client.post(
            "/api/v1/detect-raster",
            files={"file": ("sar_detect.tif", bio.getvalue(), "image/tiff")},
            data={"sensitivity": "0.35", "min_area_km2": "0.01"}
        )
        self.assertEqual(res_detect.status_code, 200)
        det_data = res_detect.json()
        self.assertIn("status", det_data)
        self.assertIn("image_type", det_data)
        self.assertIn("polygons", det_data)

        # Detect Raster endpoint with Optical Image -> must be blocked
        opt_img = Image.new("RGB", (60, 60), color=(20, 80, 180))
        opt_bio = io.BytesIO()
        opt_img.save(opt_bio, format="PNG")
        opt_bio.seek(0)

        res_opt = self.client.post(
            "/api/v1/detect-raster",
            files={"file": ("optical.png", opt_bio.getvalue(), "image/png")}
        )
        self.assertEqual(res_opt.status_code, 200)
        opt_data = res_opt.json()
        self.assertEqual(opt_data["status"], "blocked_unsupported_modality")
        self.assertFalse(opt_data["suitable_for_sar_model"])

    def test_dynamic_endpoints_drift_and_dossier(self):
        """Verifies dynamic coordinate resolution and static fallback binding."""
        # 1. Backtrack with static incident ID (INC-002 Chennai)
        res_drift = self.client.post(
            "/api/v1/drift/backtrack",
            json={"incident_id": "INC-002", "particle_count": 500, "horizon_hours": 12}
        )
        self.assertEqual(res_drift.status_code, 200)
        drift_data = res_drift.json()
        self.assertEqual(drift_data["status"], "PROCESSING")
        self.assertIn("coordinates", drift_data)
        self.assertAlmostEqual(drift_data["coordinates"]["lat"], 13.250, places=2)
        self.assertAlmostEqual(drift_data["coordinates"]["lon"], 80.460, places=2)

        # 2. Backtrack with explicit coordinate override
        res_override = self.client.post(
            "/api/v1/drift/backtrack",
            json={"incident_id": "CUSTOM-01", "lat": 12.34, "lng": 78.90, "particle_count": 100}
        )
        self.assertEqual(res_override.status_code, 200)
        override_data = res_override.json()
        self.assertAlmostEqual(override_data["coordinates"]["lat"], 12.34, places=2)
        self.assertAlmostEqual(override_data["coordinates"]["lon"], 78.90, places=2)

        # 3. Dossier generation with dynamic incident metadata
        res_dossier = self.client.get("/api/v1/dossier/INC-002")
        self.assertEqual(res_dossier.status_code, 200)
        dossier_data = res_dossier.json()
        self.assertEqual(dossier_data["status"], "SEALED")
        self.assertIn("cryptographic_verification", dossier_data)
        self.assertIn("master_sha256", dossier_data["cryptographic_verification"])


if __name__ == "__main__":
    unittest.main()
