# -*- coding: utf-8 -*-
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_ais_live_vessels_json_contract():
    """Verify GET /api/v1/ais/live-vessels returns valid application/json with vessel records."""
    response = client.get("/api/v1/ais/live-vessels?lat=18.743&lng=71.218&radius_nm=60")
    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]
    data = response.json()
    assert "vessels" in data
    assert isinstance(data["vessels"], list)
    assert len(data["vessels"]) > 0
    # Verify primary vessel attributes
    v0 = data["vessels"][0]
    assert "mmsi" in v0
    assert "name" in v0
    assert "lat" in v0
    assert "lng" in v0

def test_ais_stream_status_json_contract():
    """Verify GET /api/v1/ais/stream-status returns valid status telemetry."""
    response = client.get("/api/v1/ais/stream-status")
    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]
    data = response.json()
    assert "status" in data
    assert "total_live_vessels_tracked" in data
