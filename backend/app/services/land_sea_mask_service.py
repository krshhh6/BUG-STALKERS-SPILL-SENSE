# -*- coding: utf-8 -*-
"""
SpillSense (SIH-26143) — Geospatial Land-Sea Mask & Exclusion Service
=====================================================================
Authoritative land-sea masking engine for SAR satellite scenes.
Eliminates false oil-spill alarms on continental land, islands, coastlines,
radar shadows, and invalid/nodata scene borders.

Data Sources:
1. Natural Earth 50m Land Vector Dataset (Global, authoritative).
2. India Detailed Coastline Outline GeoJSON (Regional precision).
3. Radiometric & morphological no-data / border detector.
"""

from __future__ import annotations

import json
import os
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass

import numpy as np
import cv2
from shapely.geometry import shape, box, Polygon, MultiPolygon
from shapely.strtree import STRtree
import pyproj

GEOD = pyproj.Geod(ellps="WGS84")


@dataclass
class LandSeaMaskResult:
    ocean_mask: np.ndarray       # (H, W) uint8: 1 = Valid Ocean, 0 = Land or NoData
    land_mask: np.ndarray        # (H, W) uint8: 1 = Land, 0 = Non-land
    nodata_mask: np.ndarray      # (H, W) uint8: 1 = NoData / Invalid Border, 0 = Valid Data
    land_coverage_pct: float     # Percentage of scene covered by land
    ocean_coverage_pct: float    # Percentage of scene covered by valid ocean
    nodata_coverage_pct: float   # Percentage of scene covered by nodata
    coastal_buffer_px: int       # Buffer margin applied along coastlines
    data_source: str             # Description of authoritative data used
    status: str                  # "OK" | "WARNING_NO_BOUNDS" | "ERROR"
    details: Dict[str, Any]

    def apply_to_probability_map(self, prob_map: np.ndarray) -> np.ndarray:
        """Zeros out oil spill probability over land and nodata pixels."""
        return prob_map * self.ocean_mask.astype(prob_map.dtype)

    def apply_to_binary_mask(self, binary_mask: np.ndarray) -> np.ndarray:
        """Zeros out oil detections over land and nodata pixels."""
        return (binary_mask.astype(np.uint8) * self.ocean_mask).astype(np.uint8)


class LandSeaMaskService:
    """
    Production-grade Land-Sea exclusion service for Sentinel-1 and other SAR rasters.
    """
    _instance: Optional[LandSeaMaskService] = None

    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            # Default to backend/app/data/geospatial
            data_dir = Path(__file__).resolve().parent.parent / "data" / "geospatial"
        self.data_dir = Path(data_dir)
        self.geoms: List[Union[Polygon, MultiPolygon]] = []
        self.tree: Optional[STRtree] = None
        self.data_source_name = "None"
        self._load_authoritative_data()

    @classmethod
    def get_instance(cls) -> LandSeaMaskService:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_authoritative_data(self) -> None:
        """Loads and indexes Natural Earth and regional coastal datasets."""
        ne_file = self.data_dir / "ne_50m_land.geojson"
        india_file = Path(__file__).resolve().parent.parent.parent.parent / "frontend" / "public" / "india-outline.geojson"

        all_geoms = []
        sources = []

        if ne_file.exists():
            try:
                with open(ne_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                features = data.get("features", [])
                for feat in features:
                    geom = feat.get("geometry")
                    if geom:
                        s_geom = shape(geom)
                        if s_geom.is_valid and not s_geom.is_empty:
                            all_geoms.append(s_geom)
                        else:
                            all_geoms.append(s_geom.buffer(0))
                sources.append(f"Natural Earth 50m ({len(features)} land polygons)")
            except Exception as e:
                print(f"[LandSeaMaskService WARN] Failed loading Natural Earth 50m: {e}")

        if india_file.exists():
            try:
                with open(india_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                features = data.get("features", [])
                for feat in features:
                    geom = feat.get("geometry")
                    if geom:
                        s_geom = shape(geom)
                        if s_geom.is_valid and not s_geom.is_empty:
                            all_geoms.append(s_geom)
                        else:
                            all_geoms.append(s_geom.buffer(0))
                sources.append(f"India Coastline Outline ({len(features)} polygons)")
            except Exception as e:
                print(f"[LandSeaMaskService WARN] Failed loading India Outline: {e}")

        if all_geoms:
            self.geoms = all_geoms
            self.tree = STRtree(all_geoms)
            self.data_source_name = " + ".join(sources)
            print(f"[LandSeaMaskService] Initialized with {len(all_geoms)} polygons from {self.data_source_name}")
        else:
            self.tree = None
            self.data_source_name = "UNAVAILABLE"
            print("[LandSeaMaskService ERROR] No authoritative land dataset could be loaded!")

    def generate_ocean_mask(
        self,
        raster_shape: Tuple[int, int],
        bounds: Optional[Dict[str, float]] = None,
        transform: Optional[Union[List[float], Tuple[float, ...]]] = None,
        crs: Optional[str] = None,
        sar_raster: Optional[np.ndarray] = None,
        coastal_buffer_px: int = 2,
        coastal_buffer_m: float = 0.0,
        nodata_threshold_db: float = -45.0
    ) -> LandSeaMaskResult:
        """
        Generates an exact 2D marine ocean mask (1 = ocean, 0 = land/nodata)
        geometrically aligned to the given SAR raster grid.
        """
        h, w = raster_shape
        land_mask = np.zeros((h, w), dtype=np.uint8)
        nodata_mask = np.zeros((h, w), dtype=np.uint8)

        # 1. Detect nodata and black artificial border pixels from SAR raster
        if sar_raster is not None:
            nodata_mask = self._detect_nodata_and_borders(sar_raster, nodata_threshold_db)

        # 2. If spatial bounds or transform are provided, rasterize authoritative land polygons
        status = "OK"
        if self.tree is not None and bounds is not None:
            min_lon = bounds.get("min_lon", bounds.get("min_x", 0.0))
            min_lat = bounds.get("min_lat", bounds.get("min_y", 0.0))
            max_lon = bounds.get("max_lon", bounds.get("max_x", 0.0))
            max_lat = bounds.get("max_lat", bounds.get("max_y", 0.0))

            if max_lon > min_lon and max_lat > min_lat:
                # Query intersecting land geometries
                query_box = box(min_lon - 0.05, min_lat - 0.05, max_lon + 0.05, max_lat + 0.05)
                hit_indices = self.tree.query(query_box)

                if len(hit_indices) > 0:
                    polys_to_rasterize = []
                    for idx in hit_indices:
                        poly = self.geoms[idx]
                        try:
                            clipped = poly.intersection(query_box)
                            if not clipped.is_empty:
                                polys_to_rasterize.append(clipped)
                        except Exception:
                            polys_to_rasterize.append(poly)

                    # Rasterize into pixel coordinate system
                    land_mask = self._rasterize_polygons(
                        polys_to_rasterize,
                        raster_shape=(h, w),
                        bounds={"min_lon": min_lon, "min_lat": min_lat, "max_lon": max_lon, "max_lat": max_lat},
                        transform=transform
                    )
            else:
                status = "WARNING_INVALID_BOUNDS"
        else:
            # Fallback when image has no georeferencing (e.g. uploaded screenshot)
            if sar_raster is not None:
                land_mask = self._radiometric_texture_land_detection(sar_raster)
                status = "ESTIMATED_RADIOMETRIC_TEXTURE"
            else:
                status = "WARNING_NO_BOUNDS"

        # 3. Apply coastal buffer safety margin if requested
        buffer_px = coastal_buffer_px
        if coastal_buffer_m > 0 and bounds is not None:
            # Estimate pixel size in meters
            lat_center = (bounds.get("min_lat", 0.0) + bounds.get("max_lat", 0.0)) / 2.0
            d_lon = bounds.get("max_lon", 1.0) - bounds.get("min_lon", 0.0)
            meters_w = d_lon * 111320.0 * math.cos(math.radians(lat_center))
            m_per_px = max(1.0, meters_w / w)
            buffer_px = max(buffer_px, int(round(coastal_buffer_m / m_per_px)))

        if buffer_px > 0 and np.any(land_mask == 1):
            kernel_size = 2 * buffer_px + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            land_mask = cv2.dilate(land_mask, kernel)

        # 4. Generate final valid ocean mask:
        # A pixel is Ocean if and only if it is NOT Land AND NOT NoData
        ocean_mask = ((land_mask == 0) & (nodata_mask == 0)).astype(np.uint8)

        total_pixels = h * w
        land_pct = round(float(np.sum(land_mask == 1)) / total_pixels * 100.0, 2)
        nodata_pct = round(float(np.sum(nodata_mask == 1)) / total_pixels * 100.0, 2)
        ocean_pct = round(float(np.sum(ocean_mask == 1)) / total_pixels * 100.0, 2)

        return LandSeaMaskResult(
            ocean_mask=ocean_mask,
            land_mask=land_mask,
            nodata_mask=nodata_mask,
            land_coverage_pct=land_pct,
            ocean_coverage_pct=ocean_pct,
            nodata_coverage_pct=nodata_pct,
            coastal_buffer_px=buffer_px,
            data_source=self.data_source_name,
            status=status,
            details={
                "raster_dimensions": [w, h],
                "coastal_buffer_applied_px": buffer_px,
                "has_georeference": bounds is not None
            }
        )

    def _rasterize_polygons(
        self,
        geoms: List[Union[Polygon, MultiPolygon]],
        raster_shape: Tuple[int, int],
        bounds: Dict[str, float],
        transform: Optional[Union[List[float], Tuple[float, ...]]] = None
    ) -> np.ndarray:
        """Rasterizes Shapely polygons into an (H, W) uint8 binary mask."""
        h, w = raster_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        min_lon = bounds["min_lon"]
        min_lat = bounds["min_lat"]
        max_lon = bounds["max_lon"]
        max_lat = bounds["max_lat"]

        span_lon = max_lon - min_lon
        span_lat = max_lat - min_lat

        if span_lon <= 0 or span_lat <= 0:
            return mask

        pts_list = []

        def extract_exterior_and_interiors(poly: Polygon):
            # Exterior ring
            ext_coords = np.array(poly.exterior.coords)
            if transform and len(transform) >= 6:
                # Affine transform [dx, 0, x0, 0, -dy, y0]
                # x = x0 + col * dx => col = (x - x0) / dx
                # y = y0 - row * dy => row = (y0 - y) / dy
                dx, _, x0, _, dy_neg, y0 = transform[:6]
                dy = abs(dy_neg) if dy_neg != 0 else (span_lat / h)
                dx = dx if dx != 0 else (span_lon / w)
                cols = (ext_coords[:, 0] - x0) / dx
                rows = (y0 - ext_coords[:, 1]) / dy
            else:
                # Direct geographic linear projection
                cols = (ext_coords[:, 0] - min_lon) / span_lon * (w - 1)
                rows = (max_lat - ext_coords[:, 1]) / span_lat * (h - 1)

            pts = np.stack([cols, rows], axis=-1).astype(np.int32)
            pts_list.append(pts)

        for g in geoms:
            if isinstance(g, Polygon):
                extract_exterior_and_interiors(g)
            elif isinstance(g, MultiPolygon):
                for p in g.geoms:
                    extract_exterior_and_interiors(p)

        for pts in pts_list:
            cv2.fillPoly(mask, [pts], color=1)

        return mask

    def _detect_nodata_and_borders(self, sar_raster: np.ndarray, nodata_threshold_db: float) -> np.ndarray:
        """Detects invalid SAR backscatter, zero-padding, NaNs, and frame margins."""
        # Standardize to 2D
        if sar_raster.ndim == 3:
            img2d = sar_raster[0]
        else:
            img2d = sar_raster

        h, w = img2d.shape
        nodata = np.zeros((h, w), dtype=np.uint8)

        # 1. Non-finite values
        invalid_finite = ~np.isfinite(img2d)
        nodata[invalid_finite] = 1

        # 2. Zero-value margins (Sentinel-1 raw swath edges typically equal 0 or <= threshold)
        if np.issubdtype(img2d.dtype, np.floating):
            # In dB representation
            is_black = (img2d <= nodata_threshold_db) | (img2d == 0.0)
        else:
            # In 8-bit or 16-bit DN
            is_black = img2d <= 1

        nodata[is_black] = 1

        # 3. Outer border perimeter strip (minimum 3 pixels always flagged to prevent border convolution leakage)
        nodata[0:3, :] = 1
        nodata[h - 3:h, :] = 1
        nodata[:, 0:3] = 1
        nodata[:, w - 3:w] = 1

        return nodata

    def _radiometric_texture_land_detection(self, sar_raster: np.ndarray) -> np.ndarray:
        """
        Advanced SAR land & radar shadow discriminator for un-georeferenced images.
        Differentiates genuine smooth ocean slicks from terrestrial land, mountain radar shadows,
        and coastal urban clutter using backscatter amplitude, local variance, and layover edge peaks.
        """
        if sar_raster.ndim == 3:
            img = sar_raster[0].astype(np.float32)
        else:
            img = sar_raster.astype(np.float32)

        h, w = img.shape
        land = np.zeros((h, w), dtype=np.uint8)

        # Normalize to 0-255 if in dB or 0-1
        if np.min(img) < 0:
            # dB scale [-35 to +5]
            clipped = np.clip(img, -35.0, 5.0)
            scaled = (clipped - (-35.0)) / 40.0 * 255.0
        elif np.max(img) <= 1.01:
            scaled = img * 255.0
        else:
            scaled = np.clip(img, 0, 255)

        scaled_u8 = scaled.astype(np.uint8)

        # 1. High radar backscatter terrain & urban scatterers (> 160 in 8-bit scale / > -5 dB)
        bright_land = scaled_u8 >= 165
        land[bright_land] = 1

        # 2. Local texture variance: land has high micro-roughness compared to ocean
        blur = cv2.GaussianBlur(scaled, (7, 7), 0)
        sq_diff = (scaled - blur) ** 2
        local_var = cv2.boxFilter(sq_diff, -1, (15, 15))
        rough_terrain = (local_var > 450.0) & (scaled_u8 > 35)
        land[rough_terrain] = 1

        # 3. Radar shadow rejection:
        # Mountains cast steep radar shadows (very dark < 20) with adjacent foreshortening layover ridges (> 180)
        dark_zones = scaled_u8 <= 20
        # Check if near bright layover ridges
        kernel_ridge = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        bright_dilated = cv2.dilate((scaled_u8 >= 175).astype(np.uint8), kernel_ridge)
        radar_shadow = dark_zones & (bright_dilated == 1)
        land[radar_shadow] = 1

        # 4. Morphological closure on land mass to bridge interior gaps
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        land_closed = cv2.morphologyEx(land, cv2.MORPH_CLOSE, kernel_close)

        return land_closed
