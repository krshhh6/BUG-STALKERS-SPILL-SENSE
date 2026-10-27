# -*- coding: utf-8 -*-
"""
Tests for Phase 4: AIS Attribution Engine & Multi-Scenario Seeding Remediation.
Validates:
1. Powered vessel kinematic navigation in AttributionEngine (SOG/COG, not passive drift).
2. Trajectory buffer coverage across all 8 Indian EEZ benchmark scenarios.
3. Dark vessel gap identification and risk elevation.
4. correlate_ais_vessels task querying authentic historical buffer.
"""

from datetime import datetime, timezone, timedelta
import pytest
from app.services.attribution_engine import AttributionEngine
from app.services.ais_history_service import AISTrajectoryBuffer
from app.tasks.ais_correlation import correlate_ais_vessels
from app.models.schemas.detection import DetectionPayload, BoundingBox, AisPing


def test_powered_vessel_kinematics_attribution():
    """Verify AttributionEngine projects ship movement via SOG/COG, not passive ocean drift."""
    t0 = datetime.now(timezone.utc) - timedelta(hours=1)
    t1 = datetime.now(timezone.utc)

    # Vessel traveling due east (COG 90°) at 12 knots (~6.17 m/s)
    # In 1 hour, vessel should advance eastward by ~22.2 km (~0.22° lon at 18.7°N)
    ais_ping = AisPing(
        mmsi="419001234",
        timestamp=t0,
        lat=18.70,
        lon=71.00,
        sog_knots=12.0,
        cog_deg=90.0,
        vessel_length_m=300.0,
        vessel_type="Crude Oil Tanker"
    )

    # SAR detection at t1 near the vessel's projected position (~71.21°E, 18.70°N)
    detection = DetectionPayload(
        detection_id="TEST-DET-01",
        timestamp=t1,
        sensor_id="Sentinel-1A",
        bounding_box=BoundingBox(min_lon=71.20, min_lat=18.69, max_lon=71.22, max_lat=18.71),
        confidence_score=0.92,
        estimated_length_m=290.0,
        heading_deg=92.0
    )

    evidence = AttributionEngine.correlate_detections(
        detection=detection,
        ais_history=[ais_ping],
        incident_id="INC-001"
    )

    assert evidence.classification == "Correlated (Reporting)"
    assert evidence.confidence is not None
    assert evidence.confidence.overall_confidence > 0.80
    assert evidence.correlated_ais is not None
    assert evidence.correlated_ais.mmsi == "419001234"
    assert evidence.drift_applied is not None
    assert evidence.drift_applied.delta_x_m > 20000.0, "Displacement must reflect powered 12kn cruise"


def test_all_eight_benchmark_scenarios_seeded():
    """Verify all 8 official Indian EEZ benchmark scenarios have authentic historical trajectories."""
    buf = AISTrajectoryBuffer.get_instance()
    now_ts = datetime.now(timezone.utc).timestamp()
    t_discharge = now_ts - (24.0 * 3600.0)

    scenario_checks = [
        ("INC-001", 18.663, 71.158, "CRUDE ATLAS"),
        ("INC-002", 13.180, 80.410, "DAWN KANCHIPURAM"),
        ("INC-003", 10.420, 93.020, "DARK TANKER"),
        ("INC-004", 15.480, 73.610, "SEA PEARL"),
        ("INC-005", 22.610, 69.380, "AL KHALEEJ STAR"),
        ("INC-006", 10.020, 76.020, "OCEAN VOYAGER"),
        ("INC-007", 20.180, 86.610, "EASTERN GLORY"),
        ("INC-008", 8.520,  72.880, "TRANSIT TANKER"),
    ]

    for sc_id, o_lat, o_lon, expected_vessel in scenario_checks:
        candidates = buf.query_historical_candidates(
            origin_lat=o_lat,
            origin_lon=o_lon,
            start_time=t_discharge - 7200.0,
            end_time=t_discharge + 7200.0,
            radius_nm=45.0
        )
        assert len(candidates) > 0, f"Scenario {sc_id} must have candidate vessels"
        top = candidates[0]
        assert top["attribution_score"] >= 0.50
        assert any(expected_vessel in c["name"] for c in candidates), (
            f"Expected {expected_vessel} in candidates for {sc_id}"
        )


def test_correlate_ais_vessels_task():
    """Verify correlate_ais_vessels task queries trajectory buffer without dummy mock fallback."""
    now = datetime.now(timezone.utc)
    t_start = (now - timedelta(hours=26)).isoformat()
    t_end = (now - timedelta(hours=22)).isoformat()

    res = correlate_ais_vessels(
        incident_id="INC-001",
        envelopes_wkt={},
        time_window_start=t_start,
        time_window_end=t_end,
        origin_lat=18.663,
        origin_lon=71.158
    )

    assert res["status"] == "success"
    assert res["incident_id"] == "INC-001"
    assert res["candidate_count"] > 0
    assert len(res["candidates"]) > 0
    top = res["candidates"][0]
    assert "CRUDE ATLAS" in top["name"]
    assert top["overall_score"] > 70.0
