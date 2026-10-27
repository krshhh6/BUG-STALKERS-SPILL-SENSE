# -*- coding: utf-8 -*-
"""
Tests for Phase 2: Multi-Tiered Met-Ocean Environmental Ingestion Service.
Validates:
1. Live Open-Meteo API query with graceful fallback.
2. In-memory LRU cache retrieval.
3. Calibrated Indian EEZ regional climatology anchors (Mumbai High, Chennai, Andaman, etc.).
4. End-to-end integration with LagrangianDriftService using live/calibrated forcing provider.
"""

from datetime import datetime, timezone
import pytest
from app.services.environmental_service import EnvironmentalService, ForcingVector
from app.services.lagrangian_drift_service import LagrangianDriftService, DriftSimulationConfig


def test_calibrated_eez_regional_anchors():
    """Verify calibrated EEZ returns physical, geographically distinct vectors for Indian basins."""
    # 1. Mumbai High (18.743°N, 71.218°E)
    vec_mumbai = EnvironmentalService.get_calibrated_eez_vector(18.743, 71.218)
    assert vec_mumbai.u_current_ms > 0.20, "Mumbai High current must have positive eastward component"
    assert vec_mumbai.v_current_ms > 0.10, "Mumbai High current must have positive northward component"
    assert vec_mumbai.wave_height_m >= 1.0

    # 2. Chennai-Ennore (13.250°N, 80.460°E) - East coast northward current
    vec_chennai = EnvironmentalService.get_calibrated_eez_vector(13.250, 80.460)
    assert vec_chennai.v_current_ms > 0.25, "EICC current along Chennai must flow northward"
    assert vec_chennai.u_current_ms < vec_mumbai.u_current_ms, "Chennai zonal flow differs from Arabian Sea jet"


def test_in_memory_forcing_cache():
    """Verify in-memory caching stores and returns vectors without redundant network overhead."""
    lat, lon = 18.743, 71.218
    # Force store in cache
    key = (round(lat * 2.0) / 2.0, round(lon * 2.0) / 2.0)
    test_vec = ForcingVector(u_current_ms=0.999, v_current_ms=0.888, u_wind_ms=5.0, v_wind_ms=5.0)
    import time
    EnvironmentalService._live_cache[key] = (time.time(), test_vec)

    retrieved = EnvironmentalService.get_forcing_vector(lat, lon)
    assert retrieved.u_current_ms == 0.999
    assert retrieved.v_current_ms == 0.888


def test_live_open_meteo_or_offline_fallback():
    """Verify get_forcing_vector returns a valid ForcingVector under any network condition."""
    vec = EnvironmentalService.get_forcing_vector(18.743, 71.218)
    assert isinstance(vec, ForcingVector)
    assert -5.0 < vec.u_current_ms < 5.0
    assert -5.0 < vec.v_current_ms < 5.0
    assert -50.0 < vec.u_wind_ms < 50.0
    assert -50.0 < vec.v_wind_ms < 50.0
    assert vec.wave_height_m >= 0.0


def test_end_to_end_drift_with_environmental_provider():
    """Verify LagrangianDriftService seamlessly consumes EnvironmentalService.get_forcing_vector."""
    drift_service = LagrangianDriftService.get_instance()
    spill_lat, spill_lon = 18.743, 71.218
    t_detect = datetime.now(timezone.utc)

    config = DriftSimulationConfig(
        horizon_hours=12.0,
        time_step_sec=-600.0,
        particle_count=300,
        initial_spread_radius_m=500.0
    )

    res = drift_service.backtrack_simulation(
        incident_id="INC-001-INTEGRATION",
        spill_lat=spill_lat,
        spill_lon=spill_lon,
        detection_time=t_detect,
        forcing_provider=EnvironmentalService.get_forcing_vector,
        config=config
    )

    assert res.status == "SUCCESS"
    assert res.particle_count_valid > 250
    assert len(res.envelopes_geojson["features"]) == 3
    assert res.displacement_metrics["total_drift_distance_km"] > 0.0
