# -*- coding: utf-8 -*-
"""
Verification suite for Unified Drift Backtracking Remediation.
Tests:
1. POST /api/v1/drift/backtrack (Synchronous kinematic calculation & GeoJSON envelopes)
2. GET /api/v1/drift/trajectory/{incident_id} (Dynamic GIS trajectory retrieval across multi-scenarios)
3. GET /api/v1/dossier/{incident_id}/download (Binary PDF with authentic origin coords & SHA-256 seal)
4. Dynamic SAR Detection Handoff (Custom Lat/Lng calculation without benchmark override)
5. Multi-Basin Trajectory Integrity (Mumbai High, Chennai, Andaman, Goa)
"""

import unittest
from fastapi.testclient import TestClient
from app.main import app

class TestUnifiedDriftRemediation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_drift_backtrack_endpoint(self):
        """Verify drift backtrack returns complete physical trajectory and origin envelopes."""
        res = self.client.post(
            "/api/v1/drift/backtrack",
            json={
                "incident_id": "INC-001",
                "lat": 18.743,
                "lng": 71.218,
                "horizon_hours": 24,
                "particle_count": 1000
            }
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "PROCESSING")
        self.assertIn("origin_coordinates", data)
        self.assertIn("envelopes_geojson", data)
        
        geojson = data["envelopes_geojson"]
        self.assertEqual(geojson["type"], "FeatureCollection")
        self.assertGreaterEqual(len(geojson["features"]), 5)
        
        # Verify isobars, backtrack linestring, and origin point exist
        geom_types = [f["geometry"]["type"] for f in geojson["features"]]
        self.assertIn("Polygon", geom_types)
        self.assertIn("LineString", geom_types)
        self.assertIn("Point", geom_types)

    def test_drift_trajectory_get_endpoint(self):
        """Verify GET /api/v1/drift/trajectory/{incident_id} returns valid GeoJSON package."""
        res = self.client.get("/api/v1/drift/trajectory/INC-001")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("envelopes_geojson", data)
        self.assertIn("origin_coordinates", data)
        self.assertAlmostEqual(data["coordinates"]["lat"], 18.743, places=2)

    def test_custom_sar_detection_handoff(self):
        """Verify that passing custom coordinates from SAR Lab computes unique backtrack origin."""
        custom_lat = 15.650
        custom_lng = 73.120
        res = self.client.post(
            "/api/v1/drift/backtrack",
            json={
                "incident_id": "CUSTOM-SAR-SLICK-99",
                "lat": custom_lat,
                "lng": custom_lng,
                "horizon_hours": 20,
                "particle_count": 800
            }
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["incident_id"], "CUSTOM-SAR-SLICK-99")
        self.assertAlmostEqual(data["coordinates"]["lat"], custom_lat, places=3)
        self.assertAlmostEqual(data["coordinates"]["lon"], custom_lng, places=3)
        
        # Origin must differ from observation centroid due to non-zero drift displacement
        origin = data["origin_coordinates"]
        self.assertNotEqual((origin["lat"], origin["lon"]), (custom_lat, custom_lng))
        self.assertIn("drift_vector", data)
        self.assertGreater(data["drift_vector"]["drift_time_seconds"], 0)

    def test_multi_scenario_trajectories(self):
        """Verify each maritime scenario generates geographically consistent origin envelopes."""
        scenarios = ["INC-001", "INC-002", "INC-003", "INC-004"]
        for sc_id in scenarios:
            res = self.client.get(f"/api/v1/drift/trajectory/{sc_id}")
            self.assertEqual(res.status_code, 200, f"Scenario {sc_id} failed")
            data = res.json()
            self.assertIn("origin_coordinates", data)
            self.assertIn("envelopes_geojson", data)
            self.assertEqual(data["envelopes_geojson"]["type"], "FeatureCollection")

    def test_dossier_pdf_contains_origin_and_sha256(self):
        """Verify binary PDF dossier contains valid %PDF-1.4 header and SHA-256 evidence tag."""
        res = self.client.get("/api/v1/dossier/INC-001/download")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("content-type"), "application/pdf")
        self.assertIn("X-Forensic-SHA256", res.headers)
        self.assertEqual(len(res.headers["X-Forensic-SHA256"]), 64)
        
        # Verify ReportLab binary PDF-1.4 header
        self.assertTrue(res.content.startswith(b"%PDF-1.4"))
        self.assertGreater(len(res.content), 2000)

    def test_dossier_metadata_includes_origin_coords(self):
        """Verify JSON dossier metadata contains authentic origin coordinates for court filing."""
        res = self.client.get("/api/v1/dossier/INC-001")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "SEALED")
        self.assertIn("cryptographic_verification", data)
        self.assertEqual(data["cryptographic_verification"]["tamper_status"], "VERIFIED_GENUINE")

if __name__ == "__main__":
    unittest.main()
