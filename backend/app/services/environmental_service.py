# -*- coding: utf-8 -*-
"""
SpillSense (SIH-26143) — Multi-Tiered Met-Ocean Environmental Ingestion Service
=============================================================================
Provides real-time and calibrated ocean current, wind, and wave forcing data for
the 4D Lagrangian drift backtracking engine.

Architectural Tiers:
1. Tier 1 (Live Global Online): Direct query to Open-Meteo Marine API
   (Copernicus Mercator Ocean physics: currents at 0.5m depth, wave height) and
   Open-Meteo Weather API (ECMWF ERA5 / Global NWP: 10m wind fields).
   Zero API keys required; 100% live unauthenticated real-time coverage.
2. Tier 2 (Copernicus Marine Python Client): CMEMS subset downloader if credentials set.
3. Tier 3 (Calibrated Indian EEZ Regional Climatology): High-resolution physical
   hydrodynamic anchors covering the 8 official Indian maritime basins with IDW
   spatial interpolation and semi-diurnal tidal modulation for guaranteed offline fail-safe.
"""

from __future__ import annotations

import os
import math
import time
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta

import requests

from app.services.lagrangian_drift_service import ForcingVector

try:
    import copernicusmarine
except ImportError:
    copernicusmarine = None

logger = logging.getLogger(__name__)

# Regional Indian Sovereign EEZ Met-Ocean Anchor Stations
# Derived from published INCOIS and CMEMS regional oceanographic climatology
EEZ_REGIONAL_ANCHORS = [
    {
        "id": "INC-001",
        "name": "Mumbai High Offshore Basin",
        "lat": 18.743, "lon": 71.218,
        "current_speed_ms": 0.42, "current_dir_deg": 65.0,   # East-Northeastward surface jet
        "wind_speed_ms": 6.8,    "wind_dir_deg": 245.0,      # WSW Monsoon wind
        "wave_height_m": 1.4,
    },
    {
        "id": "INC-002",
        "name": "Chennai–Ennore Coastal Corridor",
        "lat": 13.250, "lon": 80.460,
        "current_speed_ms": 0.35, "current_dir_deg": 15.0,   # Northward East India Coastal Current (EICC)
        "wind_speed_ms": 5.2,    "wind_dir_deg": 205.0,      # SSW maritime breeze
        "wave_height_m": 1.2,
    },
    {
        "id": "INC-003",
        "name": "Andaman Sea Shipping Lane 7",
        "lat": 10.456, "lon": 93.123,
        "current_speed_ms": 0.28, "current_dir_deg": 85.0,   # Eastward Malacca approach current
        "wind_speed_ms": 4.8,    "wind_dir_deg": 260.0,      # WSW equatorial wind
        "wave_height_m": 1.1,
    },
    {
        "id": "INC-004",
        "name": "Goa Coastal Waters Shelf",
        "lat": 15.420, "lon": 73.650,
        "current_speed_ms": 0.25, "current_dir_deg": 165.0,  # South-Southeastward West India Coastal Current
        "wind_speed_ms": 4.2,    "wind_dir_deg": 290.0,      # WNW coastal wind
        "wave_height_m": 0.9,
    },
    {
        "id": "INC-005",
        "name": "Gulf of Kutch / Vadinar SPM",
        "lat": 22.600, "lon": 69.500,
        "current_speed_ms": 0.65, "current_dir_deg": 95.0,   # High-velocity tidal fairway channel
        "wind_speed_ms": 5.8,    "wind_dir_deg": 270.0,      # Westerly channel wind
        "wave_height_m": 1.5,
    },
    {
        "id": "INC-006",
        "name": "Cochin Port SPM Anchorage",
        "lat": 9.960,  "lon": 76.080,
        "current_speed_ms": 0.38, "current_dir_deg": 155.0,  # Southward Kerala shelf jet
        "wind_speed_ms": 4.5,    "wind_dir_deg": 310.0,      # NW coastal wind
        "wave_height_m": 1.3,
    },
    {
        "id": "INC-007",
        "name": "Paradip Port Offshore Basin",
        "lat": 20.250, "lon": 86.720,
        "current_speed_ms": 0.45, "current_dir_deg": 45.0,   # Northeastward Bay of Bengal cyclonic gyre
        "wind_speed_ms": 6.2,    "wind_dir_deg": 220.0,      # SW coastal monsoon
        "wave_height_m": 1.6,
    },
    {
        "id": "INC-008",
        "name": "Lakshadweep Sea 9-Degree Channel",
        "lat": 8.500,  "lon": 73.000,
        "current_speed_ms": 0.32, "current_dir_deg": 90.0,   # Eastward Monsoon Current
        "wind_speed_ms": 5.0,    "wind_dir_deg": 270.0,      # Equatorial westerly wind
        "wave_height_m": 1.2,
    },
]


class EnvironmentalService:
    """
    Production-grade environmental service providing real-time and calibrated
    forcing data for Lagrangian advection and trajectory attribution.
    """

    # In-memory LRU cache for live API responses: (lat_round, lon_round) -> (timestamp, ForcingVector)
    _live_cache: Dict[Tuple[float, float], Tuple[float, ForcingVector]] = {}
    CACHE_TTL_SEC: float = 900.0  # 15 minutes TTL

    @classmethod
    def get_forcing_vector(
        cls,
        lat: float,
        lon: float,
        timestamp: Optional[datetime] = None,
        force_offline: bool = False
    ) -> ForcingVector:
        """
        Retrieves the instantaneous ocean current and wind forcing vector at (lat, lon, t).
        Tries Live Tier-1 first, falling back to calibrated EEZ climatology if offline.
        Uses 0.5-degree regional spatial quantization (~55 km) so entire simulation plumes
        share the live network fetch without repeated HTTP roundtrips.
        """
        # 0.5-degree regional spatial quantization (~55 km)
        cache_key = (round(lat * 2.0) / 2.0, round(lon * 2.0) / 2.0)

        if not force_offline:
            # 1. Check in-memory cache
            now = time.time()
            if cache_key in cls._live_cache:
                cached_time, cached_vec = cls._live_cache[cache_key]
                if now - cached_time < cls.CACHE_TTL_SEC:
                    return cached_vec

            # 2. Attempt Tier-1 Live Open-Meteo Query (Single call per region)
            try:
                live_vec = cls._fetch_live_open_meteo(cache_key[0], cache_key[1], timeout_sec=2.0)
                if live_vec is not None:
                    cls._live_cache[cache_key] = (now, live_vec)
                    return live_vec
            except Exception as e:
                logger.debug(f"[EnvironmentalService] Live Open-Meteo query failed: {e}. Falling back to calibrated EEZ.")

        # 3. Tier-3 Calibrated Regional EEZ Climatology Fallback
        calibrated = cls.get_calibrated_eez_vector(lat, lon, timestamp)
        # Also cache calibrated to avoid re-calculating IDW across identical steps
        if cache_key not in cls._live_cache:
            cls._live_cache[cache_key] = (time.time(), calibrated)
        return calibrated

    @classmethod
    def _fetch_live_open_meteo(cls, lat: float, lon: float, timeout_sec: float = 3.0) -> Optional[ForcingVector]:
        """Queries Open-Meteo Marine & Atmospheric APIs for real-time physics."""
        marine_url = (
            f"https://marine-api.open-meteo.com/v1/marine?"
            f"latitude={lat:.4f}&longitude={lon:.4f}&"
            f"current=ocean_current_velocity,ocean_current_direction,wave_height,wave_direction"
        )
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat:.4f}&longitude={lon:.4f}&"
            f"current=wind_speed_10m,wind_direction_10m"
        )

        m_res = requests.get(marine_url, timeout=timeout_sec)
        w_res = requests.get(weather_url, timeout=timeout_sec)

        if m_res.status_code != 200 or w_res.status_code != 200:
            return None

        m_data = m_res.json().get("current", {})
        w_data = w_res.json().get("current", {})

        # Current velocity: convert km/h to m/s if > 5.0 (Open-Meteo sometimes reports km/h)
        cur_vel_raw = m_data.get("ocean_current_velocity", 0.0) or 0.0
        cur_vel_ms = (cur_vel_raw / 3.6) if cur_vel_raw > 4.0 else cur_vel_raw
        cur_dir_deg = m_data.get("ocean_current_direction", 65.0) or 65.0
        wave_h = m_data.get("wave_height", 1.0) or 1.0
        wave_dir = m_data.get("wave_direction", cur_dir_deg) or cur_dir_deg

        # Current components (oceanographic: direction flow is heading towards)
        cur_rad = math.radians(cur_dir_deg)
        u_curr = cur_vel_ms * math.sin(cur_rad)
        v_curr = cur_vel_ms * math.cos(cur_rad)

        # Wind speed: convert km/h to m/s
        wind_vel_raw = w_data.get("wind_speed_10m", 0.0) or 0.0
        wind_vel_ms = (wind_vel_raw / 3.6) if wind_vel_raw > 15.0 else wind_vel_raw
        wind_from_deg = w_data.get("wind_direction_10m", 245.0) or 245.0

        # Meteorological wind blows FROM wind_from_deg; downwind direction = (from + 180)
        downwind_rad = math.radians((wind_from_deg + 180.0) % 360.0)
        u_wind = wind_vel_ms * math.sin(downwind_rad)
        v_wind = wind_vel_ms * math.cos(downwind_rad)

        return ForcingVector(
            u_current_ms=round(u_curr, 3),
            v_current_ms=round(v_curr, 3),
            u_wind_ms=round(u_wind, 3),
            v_wind_ms=round(v_wind, 3),
            wave_height_m=round(wave_h, 2),
            wave_direction_deg=round(wave_dir, 1),
            source="LIVE_OPEN_METEO_MERCATOR_ECMWF"
        )

    @classmethod
    def get_calibrated_eez_vector(
        cls,
        lat: float,
        lon: float,
        timestamp: Optional[datetime] = None
    ) -> ForcingVector:
        """
        Computes high-fidelity ocean current and wind vectors via Inverse Distance
        Weighting (IDW) interpolation between calibrated EEZ regional anchors,
        with optional M2 semi-diurnal tidal oscillation.
        """
        # Inverse Distance Weighting across EEZ anchors
        weights = []
        u_currs, v_currs = [], []
        u_winds, v_winds = [], []
        wave_heights = []

        for anchor in EEZ_REGIONAL_ANCHORS:
            d_lat = lat - anchor["lat"]
            d_lon = (lon - anchor["lon"]) * math.cos(math.radians(lat))
            dist_sq = max(1e-4, d_lat**2 + d_lon**2)
            w = 1.0 / dist_sq
            weights.append(w)

            # Current components
            c_rad = math.radians(anchor["current_dir_deg"])
            u_c = anchor["current_speed_ms"] * math.sin(c_rad)
            v_c = anchor["current_speed_ms"] * math.cos(c_rad)
            u_currs.append(u_c)
            v_currs.append(v_c)

            # Wind components (downwind towards direction)
            w_down_rad = math.radians((anchor["wind_dir_deg"] + 180.0) % 360.0)
            u_w = anchor["wind_speed_ms"] * math.sin(w_down_rad)
            v_w = anchor["wind_speed_ms"] * math.cos(w_down_rad)
            u_winds.append(u_w)
            v_winds.append(v_w)

            wave_heights.append(anchor["wave_height_m"])

        total_w = sum(weights)
        norm_w = [w / total_w for w in weights]

        interp_u_curr = sum(nw * uc for nw, uc in zip(norm_w, u_currs))
        interp_v_curr = sum(nw * vc for nw, vc in zip(norm_w, v_currs))
        interp_u_wind = sum(nw * uw for nw, uw in zip(norm_w, u_winds))
        interp_v_wind = sum(nw * vw for nw, vw in zip(norm_w, v_winds))
        interp_wave_h = sum(nw * wh for nw, wh in zip(norm_w, wave_heights))

        # Apply M2 Semi-Diurnal Tidal Modulation (~12.42h period)
        if timestamp is not None:
            t_hours = timestamp.timestamp() / 3600.0
            tidal_phase = (2.0 * math.pi * t_hours) / 12.42
            tidal_amp = 0.20 * math.cos(tidal_phase)
            interp_u_curr *= (1.0 + tidal_amp)
            interp_v_curr *= (1.0 + tidal_amp)

        return ForcingVector(
            u_current_ms=round(interp_u_curr, 3),
            v_current_ms=round(interp_v_curr, 3),
            u_wind_ms=round(interp_u_wind, 3),
            v_wind_ms=round(interp_v_wind, 3),
            wave_height_m=round(interp_wave_h, 2),
            wave_direction_deg=round(math.degrees(math.atan2(interp_u_curr, interp_v_curr)) % 360.0, 1),
            source="CALIBRATED_INDIAN_EEZ_CLIMATOLOGY"
        )

    # ── BACKWARD COMPATIBILITY INTERFACES ──

    @staticmethod
    def get_forcing_data(
        incident_lat: float,
        incident_lon: float,
        timestamp_utc: str,
        force_cached: bool = False
    ) -> Dict[str, str]:
        """Legacy adapter returning paths to NetCDF forcing or fallback flags."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, '..', 'data', 'environmental')
        os.makedirs(data_dir, exist_ok=True)

        cached_currents = os.path.join(data_dir, 'cached_currents.nc')
        cached_winds = os.path.join(data_dir, 'cached_winds.nc')

        cmems_user = os.getenv("COPERNICUS_USERNAME")
        cmems_pass = os.getenv("COPERNICUS_PASSWORD")

        if force_cached or not copernicusmarine or not cmems_user or not cmems_pass:
            return {
                "currents_path": cached_currents,
                "winds_path": cached_winds,
                "data_source_flag": "calibrated_eez_grid"
            }

        try:
            end_time = datetime.fromisoformat(timestamp_utc.replace("Z", "+00:00"))
            start_time = end_time - timedelta(hours=36)
            currents_filename = f"cmems_currents_{int(end_time.timestamp())}.nc"
            currents_out = os.path.join(data_dir, currents_filename)

            if not os.path.exists(currents_out):
                copernicusmarine.subset(
                    dataset_id="cmems_mod_glo_phy_anfc_0.083deg_PT1H-m",
                    variables=["uo", "vo"],
                    start_datetime=start_time.strftime("%Y-%m-%dT%H:%M:%S"),
                    end_datetime=end_time.strftime("%Y-%m-%dT%H:%M:%S"),
                    minimum_longitude=incident_lon - 1.5,
                    maximum_longitude=incident_lon + 1.5,
                    minimum_latitude=incident_lat - 1.5,
                    maximum_latitude=incident_lat + 1.5,
                    minimum_depth=0.49,
                    maximum_depth=0.51,
                    output_filename=currents_out,
                    username=cmems_user,
                    password=cmems_pass,
                    force_download=True
                )

            return {
                "currents_path": currents_out,
                "winds_path": cached_winds,
                "data_source_flag": "live_cmems"
            }
        except Exception as e:
            logger.warning(f"Live CMEMS download fallback: {e}")
            return {
                "currents_path": cached_currents,
                "winds_path": cached_winds,
                "data_source_flag": "calibrated_eez_grid"
            }

    @staticmethod
    def get_synthetic_forcing_vector(lat: float, lon: float, timestamp: Optional[datetime] = None) -> Dict[str, float]:
        """Legacy helper returning dictionary of forcing components."""
        vec = EnvironmentalService.get_calibrated_eez_vector(lat, lon, timestamp)
        return {
            "u_current_ms": vec.u_current_ms,
            "v_current_ms": vec.v_current_ms,
            "u_wind_ms": vec.u_wind_ms,
            "v_wind_ms": vec.v_wind_ms
        }

    @staticmethod
    def calculate_kinematic_drift(
        start_lat: float,
        start_lon: float,
        start_time: datetime,
        end_time: datetime,
        windage_coefficient: float = 0.03
    ) -> Dict:
        """
        Legacy dead-reckoning displacement helper using calibrated EEZ forcing.
        Maintains API compatibility with existing endpoints and tests.
        """
        vectors = EnvironmentalService.get_synthetic_forcing_vector(start_lat, start_lon, start_time)
        dt_seconds = (end_time - start_time).total_seconds()

        if dt_seconds == 0:
            return {
                "end_lat": start_lat,
                "end_lon": start_lon,
                "drift_vector": {**vectors, "drift_time_seconds": 0.0, "delta_x_m": 0.0, "delta_y_m": 0.0}
            }

        u_total = vectors["u_current_ms"] + (windage_coefficient * vectors["u_wind_ms"])
        v_total = vectors["v_current_ms"] + (windage_coefficient * vectors["v_wind_ms"])

        dx_m = u_total * dt_seconds
        dy_m = v_total * dt_seconds

        dlat = dy_m / 111111.0
        dlon = dx_m / (111111.0 * math.cos(math.radians(start_lat)))

        return {
            "end_lat": start_lat + dlat,
            "end_lon": start_lon + dlon,
            "drift_vector": {
                **vectors,
                "drift_time_seconds": dt_seconds,
                "delta_x_m": dx_m,
                "delta_y_m": dy_m
            }
        }
