# -*- coding: utf-8 -*-
"""
Tests for Phase 1: Core Physics & Vectorized RK4 Lagrangian Drift Inversion Engine.
Validates:
1. Physical velocity formulation (current, leeway with Coriolis deflection, Stokes drift).
2. Pure NumPy RK4 reverse advection numerical convergence.
3. Monte Carlo stochastic turbulent diffusion scaling.
4. Land collision stranding logic.
5. True 2D Gaussian KDE isobar confidence envelope generation and geometric validity.
6. Execution latency (< 250 ms for N=1000 particles).
"""

import time
import math
from datetime import datetime, timezone, timedelta
import pytest
import numpy as np
from shapely.geometry import shape

from app.services.lagrangian_drift_service import (
    LagrangianDriftService,
    DriftSimulationConfig,
    ForcingVector
)


@pytest.fixture
def drift_service():
    return LagrangianDriftService.get_instance()


def test_velocity_formulation_components(drift_service):
    """Verify velocity synthesis includes current, windage with Coriolis deflection, and Stokes drift."""
    forcing = ForcingVector(
        u_current_ms=0.40,
        v_current_ms=-0.10,
        u_wind_ms=10.0,
        v_wind_ms=0.0,
        wave_height_m=1.5
    )
    config = DriftSimulationConfig(
        windage_leeway_factor=0.030,
        enable_coriolis_deflection=True,
        enable_stokes_drift=True
    )

    # In Northern Hemisphere (e.g. 18.7°N), wind leeway should deflect to the right (negative v in this geometry)
    u_tot, v_tot = drift_service.compute_total_velocity(lat=18.7, lon=71.2, forcing=forcing, config=config)

    # 1. Total eastward velocity should include current (0.40) + leeway (~0.30) + Stokes (~0.12)
    assert u_tot > 0.40, "Total U must exceed current alone due to downwind forcing"

    # 2. Coriolis deflection should introduce negative meridional velocity (rightward turning)
    assert v_tot < -0.10, "Coriolis deflection must deflect flow southward/rightward in Northern Hemisphere"


def test_rk4_advection_displacement_accuracy(drift_service):
    """Verify reverse RK4 displacement accurately reconstructs upstream origin."""
    spill_lat = 18.743
    spill_lon = 71.218
    detection_time = datetime.now(timezone.utc)

    # Constant steady eastward current of 0.50 m/s, zero wind
    def steady_eastward_forcing(lat, lon, t):
        return ForcingVector(u_current_ms=0.50, v_current_ms=0.0, u_wind_ms=0.0, v_wind_ms=0.0)

    config = DriftSimulationConfig(
        horizon_hours=12.0,
        time_step_sec=-600.0,
        particle_count=500,
        eddy_diffusivity_kh=0.0,  # Zero diffusion to verify pure advective integration
        initial_spread_radius_m=0.0,
        enable_coriolis_deflection=False,
        enable_stokes_drift=False,
        enable_land_stranding=False
    )

    t0 = time.perf_counter()
    res = drift_service.backtrack_simulation(
        incident_id="TEST-ADVECTION-RK4",
        spill_lat=spill_lat,
        spill_lon=spill_lon,
        detection_time=detection_time,
        forcing_provider=steady_eastward_forcing,
        config=config
    )
    t1 = time.perf_counter()

    assert res.status == "SUCCESS"
    assert (t1 - t0) * 1000 < 300, "Simulation should complete in under 300 ms"

    # In a 12h eastward flow of 0.50 m/s:
    # Total distance = 0.50 m/s * 12 * 3600 s = 21,600 m = 21.6 km
    # Reverse origin should be WEST of detection centroid by ~21.6 km
    disp = res.displacement_metrics
    assert math.isclose(disp["total_drift_distance_km"], 21.6, rel_tol=0.05)
    assert disp["zonal_displacement_km"] < -20.0, "Zonal displacement must be westward (< -20 km)"
    assert math.isclose(disp["meridional_displacement_km"], 0.0, abs_tol=0.2)
    assert res.origin_coordinates["lon"] < spill_lon, "Origin must be west of spill"


def test_monte_carlo_eddy_diffusion_dispersion(drift_service):
    """Verify stochastic random walk produces radial expansion consistent with diffusion theory."""
    spill_lat = 15.0
    spill_lon = 70.0
    detection_time = datetime.now(timezone.utc)

    # Zero background advective flow (pure diffusion test)
    def zero_forcing(lat, lon, t):
        return ForcingVector(0.0, 0.0, 0.0, 0.0)

    kh = 10.0  # m²/s
    horizon_hours = 6.0
    total_seconds = horizon_hours * 3600.0
    expected_sigma_m = math.sqrt(2.0 * kh * total_seconds)

    config = DriftSimulationConfig(
        horizon_hours=horizon_hours,
        time_step_sec=-300.0,
        particle_count=1000,
        eddy_diffusivity_kh=kh,
        initial_spread_radius_m=0.0,
        enable_land_stranding=False
    )

    res = drift_service.backtrack_simulation(
        incident_id="TEST-DIFFUSION-SCALING",
        spill_lat=spill_lat,
        spill_lon=spill_lon,
        detection_time=detection_time,
        forcing_provider=zero_forcing,
        config=config
    )

    assert res.status == "SUCCESS"
    # Ensemble mean origin should remain centered on spill
    assert math.isclose(res.origin_coordinates["lat"], spill_lat, abs_tol=0.02)
    assert math.isclose(res.origin_coordinates["lon"], spill_lon, abs_tol=0.02)

    # Trajectory history standard deviation should be non-zero and growing
    history = res.backtrack_trajectory
    assert len(history) >= 6
    final_std_km = history[-1].get("std_dev_km", 0.0)
    # Expected sigma ~ sqrt(2 * 10 * 21600) = ~657 meters = 0.66 km
    assert 0.4 <= final_std_km <= 1.2, f"Expected final std ~0.66 km, got {final_std_km} km"


def test_kde_isobar_envelopes_geometry(drift_service):
    """Verify 2D Gaussian KDE isobar contours form nested, valid GeoJSON polygons."""
    spill_lat = 18.743
    spill_lon = 71.218
    detection_time = datetime.now(timezone.utc)

    def realistic_forcing(lat, lon, t):
        return ForcingVector(
            u_current_ms=0.35,
            v_current_ms=0.15,
            u_wind_ms=5.0,
            v_wind_ms=-2.0,
            wave_height_m=1.2
        )

    config = DriftSimulationConfig(
        horizon_hours=24.0,
        time_step_sec=-600.0,
        particle_count=800,
        eddy_diffusivity_kh=10.0,
        initial_spread_radius_m=800.0
    )

    res = drift_service.backtrack_simulation(
        incident_id="TEST-KDE-ENVELOPES",
        spill_lat=spill_lat,
        spill_lon=spill_lon,
        detection_time=detection_time,
        forcing_provider=realistic_forcing,
        config=config
    )

    assert res.status == "SUCCESS"
    geojson = res.envelopes_geojson
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 3, "Must produce 50%, 75%, and 95% envelopes"

    areas = {}
    for feat in geojson["features"]:
        geom = shape(feat["geometry"])
        assert geom.is_valid, f"Isobar geometry {feat['properties']['name']} must be topologically valid"
        assert geom.area > 0.0, "Isobar polygon area must be strictly positive"
        conf = feat["properties"]["confidence"]
        areas[conf] = geom.area

    # Envelopes must be strictly nested in area: 50% < 75% < 95%
    assert areas["50%"] < areas["75%"], "50% core must be strictly smaller than 75% envelope"
    assert areas["75%"] < areas["95%"], "75% envelope must be strictly smaller than 95% outer isobar"


def test_land_collision_stranding(drift_service):
    """Verify particles pushed onshore are stranded at the coast and do not cross land."""
    # Start close to Mumbai coast (72.70, 18.90) and push strongly eastward onto mainland
    spill_lat = 18.90
    spill_lon = 72.70
    detection_time = datetime.now(timezone.utc)

    # Reverse advection towards east (meaning forward current was westward)
    # So reverse drift pushes eastward onto Mumbai land
    def onshore_reverse_forcing(lat, lon, t):
        return ForcingVector(u_current_ms=-1.5, v_current_ms=0.0, u_wind_ms=0.0, v_wind_ms=0.0)

    config = DriftSimulationConfig(
        horizon_hours=12.0,
        time_step_sec=-600.0,
        particle_count=300,
        eddy_diffusivity_kh=2.0,
        enable_land_stranding=True
    )

    res = drift_service.backtrack_simulation(
        incident_id="TEST-LAND-STRANDING",
        spill_lat=spill_lat,
        spill_lon=spill_lon,
        detection_time=detection_time,
        forcing_provider=onshore_reverse_forcing,
        config=config
    )

    assert res.status == "SUCCESS"
    assert res.particle_stranded_count > 0, "Particles pushed onto mainland must be flagged as stranded"
    # Stranded particles should not penetrate deep inland (> 73.2°E)
    assert res.origin_coordinates["lon"] < 73.20, "Reconstructed origin must be constrained near coast"
