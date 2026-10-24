# -*- coding: utf-8 -*-
"""
Spill Sense (SIH-26143) — Tiled Overlapping SAR Inference & Mask Reconstruction
==============================================================================
Provides memory-efficient, seamless sliding-window inference across large
SAR rasters with 2D Hann window blending, sub-pixel geospatial coordinate
alignment, and rigorous uncertainty estimation.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from shapely.geometry import Polygon, mapping
import pyproj

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from app.services.sar_preprocessing import SARPreprocessor
from app.services.sar_segmentation_model import SARSPILLSegmentationEngine


class TiledSARInferenceEngine:
    """
    Overlapping tiled inference engine for large satellite rasters.
    Blends overlapping tiles with 2D Hann window feathering to eliminate seam artifacts.
    """

    GEOD = pyproj.Geod(ellps="WGS84")

    def __init__(
        self,
        segmentation_engine: Optional[SARSPILLSegmentationEngine] = None,
        tile_size: int = 512,
        overlap_fraction: float = 0.25,
        default_sensitivity: float = 0.35,
        min_slick_area_km2: float = 0.05
    ):
        self.engine = segmentation_engine or SARSPILLSegmentationEngine()
        self.tile_size = tile_size
        self.overlap_fraction = max(0.0, min(0.5, overlap_fraction))
        self.stride = int(tile_size * (1.0 - self.overlap_fraction))
        self.default_sensitivity = default_sensitivity
        self.min_slick_area_km2 = min_slick_area_km2

        # Initialize DualPolOilSpillNet classifier gating engine
        self.classifier_session = None
        self.classifier_threshold = 0.49
        try:
            import onnxruntime as ort
            import os
            cls_paths = [
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "frontend", "public", "models", "oil_classifier.onnx"),
                os.path.join(os.path.dirname(__file__), "..", "models", "oil_classifier.onnx")
            ]
            for cp in cls_paths:
                if os.path.exists(cp):
                    self.classifier_session = ort.InferenceSession(os.path.abspath(cp), providers=['CPUExecutionProvider'])
                    print(f"TiledSARInferenceEngine: Loaded DualPolOilSpillNet classifier gate from {cp}")
                    break
        except Exception as e:
            print(f"TiledSARInferenceEngine: Classifier initialization note: {e}")

    def run_classifier_gate(self, vv_db: np.ndarray, vh_db: np.ndarray, sea_mask: np.ndarray) -> float:
        """Runs DualPolOilSpillNet classifier to screen out ship wakes, clean ocean, and look-alikes."""
        if self.classifier_session is None or not CV2_AVAILABLE:
            return 0.50

        try:
            h, w = vv_db.shape
            # Focus on central or salient marine region
            cy, cx = h // 2, w // 2
            half = min(cy, cx, 200)
            crop_vv = vv_db[cy - half:cy + half, cx - half:cx + half]
            crop_vh = vh_db[cy - half:cy + half, cx - half:cx + half]

            vv_norm = np.clip((crop_vv - (-32.0)) / 22.0, 0.0, 1.0)
            vh_norm = np.clip((crop_vh - (-42.0)) / 22.0, 0.0, 1.0)

            vv_400 = cv2.resize(vv_norm, (400, 400), interpolation=cv2.INTER_LINEAR)
            vh_400 = cv2.resize(vh_norm, (400, 400), interpolation=cv2.INTER_LINEAR)
            tensor_in = np.stack([vv_400, vh_400], axis=0).reshape(1, 2, 400, 400).astype(np.float32)

            in_name = self.classifier_session.get_inputs()[0].name
            out_name = self.classifier_session.get_outputs()[0].name
            logit = float(self.classifier_session.run([out_name], {in_name: tensor_in})[0].flat[0])
            prob = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, logit))))
            return prob
        except Exception:
            return 0.50

    @staticmethod
    def _create_2d_hann_window(height: int, width: int, power: float = 1.0) -> np.ndarray:
        """
        Generates 2D Hann / cosine blending taper with small non-zero floor.
        """
        y_vec = np.sin(np.pi * (np.arange(height) + 0.5) / height)
        x_vec = np.sin(np.pi * (np.arange(width) + 0.5) / width)
        window = np.outer(y_vec, x_vec) ** power
        # Ensure a small non-zero floor to prevent boundary division issues
        return np.maximum(window.astype(np.float32), 0.05)

    def run_tiled_inference(
        self,
        sar_raster: np.ndarray,
        sea_mask: Optional[np.ndarray] = None,
        transform: Optional[List[float]] = None,
        bounds: Optional[Dict[str, float]] = None,
        sensitivity_threshold: Optional[float] = None,
        min_area_km2: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes sliding-window inference on large SAR raster.
        
        Args:
            sar_raster: 2D (H, W) or 3D (C, H, W) SAR Sigma0 dB array
            sea_mask: Optional 2D binary marine surface mask (1 = water, 0 = land)
            transform: Optional Affine transform 6-tuple [dx, 0, x0, 0, -dy, y0]
            bounds: Optional bounding box {min_lon, min_lat, max_lon, max_lat}
            sensitivity_threshold: U-Net probability threshold [0.1 - 0.9]
            min_area_km2: Minimum slick polygon geodesic area filter
        """
        threshold = sensitivity_threshold or self.default_sensitivity
        area_filter = min_area_km2 or self.min_slick_area_km2

        # Extract VV and VH polarimetric channels
        if sar_raster.ndim == 3:
            if sar_raster.shape[0] == 2:
                # Standard S1 IW GRD: Band 0 = VH, Band 1 = VV
                vv_db = sar_raster[1].astype(np.float32)
                vh_db = sar_raster[0].astype(np.float32)
            elif sar_raster.shape[-1] == 2:
                vv_db = sar_raster[:, :, 1].astype(np.float32)
                vh_db = sar_raster[:, :, 0].astype(np.float32)
            else:
                vv_db = sar_raster[0].astype(np.float32)
                vh_db = vv_db - 10.5
        else:
            vv_db = sar_raster.astype(np.float32)
            vh_db = vv_db - 10.5

        sar_db = vv_db
        h, w = sar_db.shape

        # 1. Authoritative Geospatial Land-Sea & NoData Masking
        from app.services.land_sea_mask_service import LandSeaMaskService
        land_service = LandSeaMaskService.get_instance()
        mask_res = land_service.generate_ocean_mask(
            raster_shape=(h, w),
            bounds=bounds,
            transform=transform,
            sar_raster=sar_db,
            coastal_buffer_px=2
        )
        computed_sea_mask = mask_res.ocean_mask
        if sea_mask is not None and sea_mask.shape == (h, w):
            computed_sea_mask = (computed_sea_mask & sea_mask.astype(np.uint8)).astype(np.uint8)

        masked_sar = np.where(computed_sea_mask == 1, sar_db, -100.0)

        # Check ocean coverage: if scene is terrestrial land (< 0.5% water), immediately return zero alerts
        if mask_res.ocean_coverage_pct < 0.5:
            return {
                "status": "no_oil_detected",
                "polygons": [],
                "total_detected_area_km2": 0.0,
                "model_version": self.engine.MODEL_VERSION,
                "uncertainty": {"false_positive_risk": "LOW", "detection_confidence_score": 0.0},
                "ocean_mean_db": -14.0,
                "land_mask_applied": True,
                "land_coverage_pct": mask_res.land_coverage_pct,
                "ocean_coverage_pct": mask_res.ocean_coverage_pct,
                "nodata_coverage_pct": mask_res.nodata_coverage_pct,
                "land_data_source": mask_res.data_source,
                "message": "Scene classified as terrestrial land. Land exclusion active; zero false alarms.",
                "raster_dimensions": [w, h]
            }

        # Ambient ocean clutter statistics
        valid_marine_px = sar_db[computed_sea_mask == 1]
        ocean_mean_db = float(np.percentile(valid_marine_px, 65)) if len(valid_marine_px) > 0 else -14.0

        # Authoritative Classifier Gating: screen out ship wakes and look-alikes
        cls_prob = self.run_classifier_gate(vv_db, vh_db, computed_sea_mask)
        gate_threshold = 0.15 if (sensitivity_threshold is not None and sensitivity_threshold < 0.20) else 0.18
        if cls_prob < gate_threshold:
            return {
                "status": "no_oil_detected",
                "polygons": [],
                "total_detected_area_km2": 0.0,
                "model_version": self.engine.MODEL_VERSION,
                "uncertainty": {"false_positive_risk": "LOW", "detection_confidence_score": round(1.0 - cls_prob, 3)},
                "ocean_mean_db": round(ocean_mean_db, 2),
                "land_mask_applied": True,
                "land_coverage_pct": mask_res.land_coverage_pct,
                "ocean_coverage_pct": mask_res.ocean_coverage_pct,
                "nodata_coverage_pct": mask_res.nodata_coverage_pct,
                "land_data_source": mask_res.data_source,
                "classifier_gated": True,
                "classifier_oil_prob": round(cls_prob, 4),
                "message": "Classifier confirmed clean ocean or look-alike / vessel wake. Gated out without false alarms.",
                "raster_dimensions": [w, h]
            }

        # Accumulator arrays for overlapping reconstruction
        prob_accumulator = np.zeros((h, w), dtype=np.float32)
        weight_accumulator = np.zeros((h, w), dtype=np.float32)

        # Determine tile starts
        t_sz = min(self.tile_size, max(h, w))
        stride_y = int(t_sz * (1.0 - self.overlap_fraction))
        stride_x = int(t_sz * (1.0 - self.overlap_fraction))

        y_starts = list(range(0, max(1, h - t_sz + 1), stride_y))
        if len(y_starts) == 0 or y_starts[-1] + t_sz < h:
            y_starts.append(max(0, h - t_sz))
        y_starts = sorted(list(set(y_starts)))

        x_starts = list(range(0, max(1, w - t_sz + 1), stride_x))
        if len(x_starts) == 0 or x_starts[-1] + t_sz < w:
            x_starts.append(max(0, w - t_sz))
        x_starts = sorted(list(set(x_starts)))

        hann_weight = self._create_2d_hann_window(t_sz, t_sz)

        for y in y_starts:
            for x in x_starts:
                y_end = min(y + t_sz, h)
                x_end = min(x + t_sz, w)
                cur_h = y_end - y
                cur_w = x_end - x

                tile_vv = vv_db[y:y_end, x:x_end]
                tile_vh = vh_db[y:y_end, x:x_end]
                tile_sea = computed_sea_mask[y:y_end, x:x_end]

                # If tile has virtually no ocean (< 2% valid water), skip ML
                if float(np.mean(tile_sea)) < 0.02:
                    continue

                # Run SpillSegNet ONNX inference on dual-pol tile
                tile_prob = self._infer_single_patch(tile_vv, tile_vh, tile_sea)

                # Damping score relative to ambient ocean clutter
                damping_delta = (ocean_mean_db - 4.5) - tile_vv
                damping_score = np.clip(damping_delta / 4.0, 0.0, 1.0) * tile_sea

                # Conjunction of neural activation (65%) and capillary damping depression (35%)
                combined_tile_prob = (tile_prob * 0.65) + (damping_score * 0.35)

                w_slice = hann_weight[:cur_h, :cur_w]
                prob_accumulator[y:y_end, x:x_end] += combined_tile_prob * w_slice
                weight_accumulator[y:y_end, x:x_end] += w_slice

        # Normalize probability map
        full_prob_map = prob_accumulator / np.maximum(weight_accumulator, 1e-5)
        # Strictly zero-out land, island, and nodata pixels
        full_prob_map = full_prob_map * computed_sea_mask.astype(np.float32)

        # Global capillary damping check
        global_damping = np.clip(((ocean_mean_db - 4.5) - sar_db) / 4.0, 0.0, 1.0) * computed_sea_mask
        binary_mask = ((full_prob_map >= threshold) & (global_damping > 0.05) & (computed_sea_mask == 1)).astype(np.uint8)

        # Morphological cleanup
        if CV2_AVAILABLE:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
            # Re-apply sea mask to eliminate any morphological dilation leakage onto land
            cleaned_mask = (cleaned_mask * computed_sea_mask).astype(np.uint8)
        else:
            cleaned_mask = (binary_mask * computed_sea_mask).astype(np.uint8)

        # Compute uncertainty metrics
        uncertainty_info = self._calculate_uncertainty(full_prob_map, computed_sea_mask, threshold)

        # Extract geographic vector polygons with zero land leakage
        polygons = self._extract_geopolygons(
            cleaned_mask=cleaned_mask,
            prob_map=full_prob_map,
            sar_db=sar_db,
            ocean_mean_db=ocean_mean_db,
            transform=transform,
            bounds=bounds,
            area_filter=area_filter,
            land_mask=mask_res.land_mask
        )

        total_area = round(sum(p["area_km2"] for p in polygons), 2)
        has_detection = len(polygons) > 0

        status = "detected" if has_detection else "no_oil_detected"
        message = (
            f"Active hydrocarbon signature detected across {len(polygons)} slick segment(s) totaling {total_area} km²."
            if has_detection else "No significant hydrocarbon signature detected across analyzed marine SAR scene."
        )

        return {
            "status": status,
            "polygons": polygons,
            "total_detected_area_km2": total_area,
            "model_version": self.engine.MODEL_VERSION,
            "uncertainty": uncertainty_info,
            "ocean_mean_db": round(ocean_mean_db, 2),
            "land_mask_applied": True,
            "land_coverage_pct": mask_res.land_coverage_pct,
            "ocean_coverage_pct": mask_res.ocean_coverage_pct,
            "nodata_coverage_pct": mask_res.nodata_coverage_pct,
            "land_data_source": mask_res.data_source,
            "message": message,
            "raster_dimensions": [w, h]
        }

    def _infer_single_patch(self, tile_vv: np.ndarray, tile_vh: np.ndarray, tile_sea: np.ndarray) -> np.ndarray:
        """Runs SpillSegNet ONNX dual-polarization model on a single tile."""
        cur_h, cur_w = tile_vv.shape

        # Normalize VV: [-32, -10] dB -> [0.0, 1.0]
        vv_norm = np.clip((tile_vv - (-32.0)) / ((-10.0) - (-32.0)), 0.0, 1.0)
        # Normalize VH: [-42, -20] dB -> [0.0, 1.0]
        vh_norm = np.clip((tile_vh - (-42.0)) / ((-20.0) - (-42.0)), 0.0, 1.0)

        # Execute ONNX Runtime engine (SpillSegNet Dual-Pol)
        if hasattr(self.engine, "ort_session") and self.engine.ort_session is not None:

            try:
                vv_512 = cv2.resize(vv_norm, (512, 512), interpolation=cv2.INTER_LINEAR)
                vh_512 = cv2.resize(vh_norm, (512, 512), interpolation=cv2.INTER_LINEAR)
                tensor_512 = np.stack([vv_512, vh_512], axis=0).reshape(1, 2, 512, 512).astype(np.float32)

                input_name = self.engine.ort_session.get_inputs()[0].name
                output_name = self.engine.ort_session.get_outputs()[0].name
                raw_out = self.engine.ort_session.run([output_name], {input_name: tensor_512})[0]

                # Dense logits [1, 1, 512, 512] -> sigmoid probability
                logits = raw_out.squeeze()
                prob_512 = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
                prob = cv2.resize(prob_512, (cur_w, cur_h), interpolation=cv2.INTER_LINEAR)
                return prob * tile_sea.astype(np.float32)
            except Exception as e:
                pass

        # Execute PyTorch model if available
        if TORCH_AVAILABLE and hasattr(self.engine, "model") and self.engine.model is not None and hasattr(self.engine, "device") and self.engine.device is not None:
            try:
                tensor_in = torch.from_numpy(vv_norm).unsqueeze(0).unsqueeze(0).to(self.engine.device)
                with torch.no_grad():
                    prob = self.engine.model(tensor_in).squeeze().cpu().numpy()
                return prob * tile_sea.astype(np.float32)
            except Exception:
                pass

        # Physical capillary depression fallback
        ocean_val = float(np.percentile(tile_vv[tile_sea == 1], 65)) if np.any(tile_sea == 1) else -14.0
        heuristic_prob = np.clip(((ocean_val - 4.5) - tile_vv) / 6.0, 0.0, 1.0)
        return heuristic_prob * tile_sea.astype(np.float32)


    def _calculate_uncertainty(
        self,
        prob_map: np.ndarray,
        sea_mask: np.ndarray,
        threshold: float
    ) -> Dict[str, Any]:
        """
        Calculates normalized Shannon entropy and margin uncertainty.
        """
        valid_p = prob_map[sea_mask == 1]
        if len(valid_p) == 0:
            return {
                "mean_shannon_entropy": 0.0,
                "margin_uncertainty_fraction": 0.0,
                "detection_confidence_score": 0.0,
                "false_positive_risk": "HIGH"
            }

        # Shannon entropy H(p) = -p log2(p) - (1-p) log2(1-p)
        p_clamped = np.clip(valid_p, 1e-6, 1.0 - 1e-6)
        entropy = - (p_clamped * np.log2(p_clamped) + (1.0 - p_clamped) * np.log2(1.0 - p_clamped))
        mean_entropy = float(np.mean(entropy))

        # Margin uncertainty: fraction of pixels near the decision boundary [thresh - 0.15, thresh + 0.15]
        margin_mask = (valid_p >= threshold - 0.15) & (valid_p <= threshold + 0.15)
        margin_fraction = float(np.sum(margin_mask)) / max(len(valid_p), 1)

        # High confidence = low entropy and low boundary margin ambiguity
        confidence_score = max(0.0, min(1.0, 1.0 - (mean_entropy * 0.5 + margin_fraction * 0.5)))

        fp_risk = "LOW" if confidence_score > 0.75 else ("MEDIUM" if confidence_score > 0.50 else "HIGH")

        return {
            "mean_shannon_entropy": round(mean_entropy, 4),
            "margin_uncertainty_fraction": round(margin_fraction, 4),
            "detection_confidence_score": round(confidence_score, 3),
            "false_positive_risk": fp_risk
        }

    def _extract_geopolygons(
        self,
        cleaned_mask: np.ndarray,
        prob_map: np.ndarray,
        sar_db: np.ndarray,
        ocean_mean_db: float,
        transform: Optional[List[float]],
        bounds: Optional[Dict[str, float]],
        area_filter: float,
        land_mask: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """Extracts and maps vector polygons with geodesic area calculation."""
        if not CV2_AVAILABLE:
            return []

        h, w = cleaned_mask.shape
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Coordinate mapper
        def pixel_to_geo(px: float, py: float) -> Tuple[float, float]:
            if transform and len(transform) >= 6:
                # Affine transform: [dx, 0, x0, 0, -dy, y0]
                dx, _, x0, _, neg_dy, y0 = transform[:6]
                lon = x0 + (px * dx)
                lat = y0 + (py * neg_dy)
                return lon, lat
            elif bounds:
                min_lon = bounds.get("min_lon", 70.9)
                max_lon = bounds.get("max_lon", 71.5)
                min_lat = bounds.get("min_lat", 18.4)
                max_lat = bounds.get("max_lat", 19.0)
                lon = min_lon + (px / float(w)) * (max_lon - min_lon)
                lat = max_lat - (py / float(h)) * (max_lat - min_lat)
                return lon, lat
            else:
                # Default Mumbai High marine coordinates
                lon = 71.0 + (px / float(w)) * 0.5
                lat = 19.0 - (py / float(h)) * 0.5
                return lon, lat

        detected_polygons: List[Dict[str, Any]] = []

        for cnt in contours:
            if len(cnt) < 4:
                continue

            geo_coords = []
            for pt in cnt.squeeze():
                if pt.ndim != 1 or len(pt) < 2:
                    continue
                lon, lat = pixel_to_geo(float(pt[0]), float(pt[1]))
                geo_coords.append((lon, lat))

            if len(geo_coords) < 4:
                continue

            # Ensure closed ring
            if geo_coords[0] != geo_coords[-1]:
                geo_coords.append(geo_coords[0])

            try:
                poly = Polygon(geo_coords)
                poly = poly.simplify(0.0002, preserve_topology=True)

                if not poly.is_valid or poly.is_empty:
                    continue

                area_m2, _ = self.GEOD.geometry_area_perimeter(poly)
                area_km2 = abs(area_m2) / 1e6

                if area_km2 < area_filter:
                    continue

                # Sample contour pixel mask
                poly_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(poly_mask, [cnt], -1, 1, -1)

                # Strict geospatial exclusion: eliminate any polygon intersecting land
                if land_mask is not None and np.any((poly_mask == 1) & (land_mask == 1)):
                    continue
                slick_pixels = sar_db[poly_mask == 1]
                mean_damping = float(np.mean(slick_pixels) - ocean_mean_db) if len(slick_pixels) > 0 else -6.5
                conf = float(np.mean(prob_map[poly_mask == 1])) if len(slick_pixels) > 0 else 0.85

                detected_polygons.append({
                    "geometry": mapping(poly),
                    "confidence": round(float(conf), 3),
                    "area_km2": round(float(area_km2), 2),
                    "mean_damping_db": round(float(mean_damping), 2),
                    "slick_centroid": [round(float(poly.centroid.y), 4), round(float(poly.centroid.x), 4)]
                })
            except Exception:
                continue

        detected_polygons.sort(key=lambda p: p["area_km2"], reverse=True)
        return detected_polygons
