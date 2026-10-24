# -*- coding: utf-8 -*-
"""
Tests for AIS Historical Trajectory Buffer and 4D Spatio-Temporal Intercept Engine.
Validates:
1. Kinematic downsampling logic (linear pings rejected, turns/speed changes accepted).
2. Rolling buffer storage and retrieval.
3. 4D spatio-temporal intercept queries matching the backward Lagrangian drift origin.
4. Dark vessel (AIS blackout gap) identification and dead-reckoning risk elevation.
"""

import pytest
import os
import time
from datetime import datetime, timezone
from app.services.ais_history_service import KinematicFilter, AISTrajectoryBuffer, haversine_distance_nm

def test_kinematic_downsampler_filtering():
    """Verify that redundant linear pings are rejected while significant maneuvers are stored."""
    kf = KinematicFilter(min_interval_sec=180.0, min_cog_deg=5.0, min_sog_kn=1.5)
    t0 = 1000000.0
    mmsi = "419001234"

    # 1. Initial ping: should store
    assert kf.should_store(mmsi, lat=18.5, lon=71.0, cog=180.0, sog=14.0, timestamp=t0) is True

    # 2. Sub-second burst noise: must discard
    assert kf.should_store(mmsi, lat=18.501, lon=71.0, cog=180.0, sog=14.0, timestamp=t0 + 2.0) is False

    # 3. 60 seconds later, same course and speed: must discard
    assert kf.should_store(mmsi, lat=18.510, lon=71.0, cog=180.2, sog=14.1, timestamp=t0 + 60.0) is False

    # 4. 80 seconds later, significant turn (180° -> 188°: delta 8° >= 5°): should store
    assert kf.should_store(mmsi, lat=18.515, lon=71.002, cog=188.0, sog=14.0, timestamp=t0 + 80.0) is True

    # 5. 30 seconds later, significant deceleration (14.0 kn -> 12.0 kn: delta 2.0 >= 1.5): should store
    assert kf.should_store(mmsi, lat=18.520, lon=71.005, cog=188.0, sog=12.0, timestamp=t0 + 110.0) is True

    # 6. 200 seconds later, no change in course/speed: time threshold (>= 180s) triggered -> should store
    assert kf.should_store(mmsi, lat=18.550, lon=71.010, cog=188.0, sog=12.0, timestamp=t0 + 320.0) is True


def test_trajectory_buffer_4d_query(tmp_path):
    """Verify that 4D spatio-temporal query extracts authentic candidates intersecting the origin envelope."""
    test_db = os.path.join(tmp_path, "test_ais_buffer.db")
    buf = AISTrajectoryBuffer(db_path=test_db)

    now_ts = datetime.now(timezone.utc).timestamp()
    t_discharge = now_ts - (24.0 * 3600.0)

    # Query around Mumbai High origin: ~18.663°N, 71.158°E at T - 24h
    candidates = buf.query_historical_candidates(
        origin_lat=18.663,
        origin_lon=71.158,
        start_time=t_discharge - 7200.0,
        end_time=t_discharge + 7200.0,
        radius_nm=30.0
    )

    assert len(candidates) > 0

    # Top candidate should be CRUDE ATLAS (VLCC Tanker with dark gap over the origin)
    top_suspect = candidates[0]
    assert top_suspect["mmsi"] == "419001234"
    assert "CRUDE ATLAS" in top_suspect["name"]
    assert top_suspect["risk"] in ["CRITICAL", "HIGH"]
    assert top_suspect["attribution_score"] >= 0.70
    assert top_suspect["dark_vessel_suspect"] is True
    assert top_suspect["ais_gap_hours"] >= 4.0
    assert len(top_suspect["trajectory"]) >= 4

    # Check metrics breakdown
    metrics = top_suspect["metrics"]
    assert "spatial_match_pct" in metrics
    assert "temporal_alignment_pct" in metrics
    assert "dark_gap_suspicion_pct" in metrics
    assert "vessel_risk_prior_pct" in metrics


def test_dark_vessel_gap_dead_reckoning(tmp_path):
    """Verify that a ship with an AIS silence gap directly across the origin is detected as dark vessel suspect."""
    test_db = os.path.join(tmp_path, "test_gap_buffer.db")
    buf = AISTrajectoryBuffer(db_path=test_db)

    t0 = 2000000.0
    # Add a vessel that went dark 2 hours before the origin and reappeared 2 hours after
    # Last ping at (18.80, 71.15), reappears at (18.50, 71.15) — straight line passes through origin (18.65, 71.15)
    buf.ingest_packet(
        mmsi="999001122", name="ROGUE VOYAGER", imo="9991234", flag="Panama",
        ship_type="Crude Oil Tanker", is_tanker=True,
        lat=18.80, lon=71.15, sog=12.0, cog=180.0, recorded_at=t0 - 7200.0
    )
    buf.ingest_packet(
        mmsi="999001122", name="ROGUE VOYAGER", imo="9991234", flag="Panama",
        ship_type="Crude Oil Tanker", is_tanker=True,
        lat=18.50, lon=71.15, sog=12.0, cog=180.0, recorded_at=t0 + 7200.0
    )

    candidates = buf.query_historical_candidates(
        origin_lat=18.65,
        origin_lon=71.15,
        start_time=t0 - 3600.0,
        end_time=t0 + 3600.0,
        radius_nm=20.0
    )

    rogue = next((c for c in candidates if c["mmsi"] == "999001122"), None)
    assert rogue is not None
    assert rogue["dark_vessel_suspect"] is True
    assert rogue["ais_gap_hours"] >= 3.9
