# -*- coding: utf-8 -*-
"""
SpillSense (SIH-26143) — AIS Vessel Correlation & Forensic Attribution Task
===========================================================================
Correlates candidate vessel trajectories against the 4D Lagrangian origin
envelope reconstructed by the backward drift engine.

Evaluates:
1. Spatial Overlap (Intersection with 50%, 75%, 95% probability isobars).
2. Closest Point of Approach (CPA) and Time of Closest Approach (TCA).
3. AIS Transponder Silence (Dark vessel gap analysis during transit).
4. Vessel Risk Prior (Tanker / Bulk crude carrier vs innocent bystander).
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from celery import shared_task
from shapely.wkt import loads as wkt_loads
from shapely.geometry import Point, MultiPolygon, Polygon

from app.services.ais_history_service import AISTrajectoryBuffer

logger = logging.getLogger(__name__)


@shared_task
def correlate_ais_vessels(
    incident_id: str,
    envelopes_wkt: Dict[str, str],
    time_window_start: str,
    time_window_end: str,
    origin_lat: Optional[float] = None,
    origin_lon: Optional[float] = None
) -> Dict[str, Any]:
    """
    Correlates AIS vessel tracks against the Monte Carlo drift probability envelopes.
    Identifies candidate vessels and calculates an authentic attribution score.
    """
    try:
        buffer = AISTrajectoryBuffer.get_instance()

        # Parse temporal window
        t_start_dt = datetime.fromisoformat(time_window_start.replace("Z", "+00:00"))
        t_end_dt = datetime.fromisoformat(time_window_end.replace("Z", "+00:00"))
        t_start_ts = t_start_dt.timestamp()
        t_end_ts = t_end_dt.timestamp()

        # Parse envelopes or extract centroid
        poly_high = None
        poly_med = None
        poly_low = None

        if "high" in envelopes_wkt:
            try:
                poly_high = wkt_loads(envelopes_wkt["high"])
            except Exception:
                pass
        if "medium" in envelopes_wkt:
            try:
                poly_med = wkt_loads(envelopes_wkt["medium"])
            except Exception:
                pass
        if "low" in envelopes_wkt:
            try:
                poly_low = wkt_loads(envelopes_wkt["low"])
            except Exception:
                pass

        # If origin lat/lon not explicitly passed, infer from envelope centroid
        if origin_lat is None or origin_lon is None:
            if poly_high and not poly_high.is_empty:
                origin_lat = poly_high.centroid.y
                origin_lon = poly_high.centroid.x
            elif poly_med and not poly_med.is_empty:
                origin_lat = poly_med.centroid.y
                origin_lon = poly_med.centroid.x
            else:
                origin_lat = 18.743
                origin_lon = 71.218

        # Query 4D historical trajectory buffer
        candidates_raw = buffer.query_historical_candidates(
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            start_time=t_start_ts,
            end_time=t_end_ts,
            radius_nm=40.0
        )

        # Refine candidates with exact polygon intersection if WKT polygons exist
        candidates = []
        for c in candidates_raw:
            # Check if any waypoint or trajectory segment physically intersected the polygon envelopes
            hit_high = False
            hit_med = False
            hit_low = False

            if poly_high or poly_med or poly_low:
                for pt in c.get("trajectory", []):
                    p = Point(pt["lng"], pt["lat"])
                    if poly_high and poly_high.contains(p):
                        hit_high = True
                    if poly_med and poly_med.contains(p):
                        hit_med = True
                    if poly_low and poly_low.contains(p):
                        hit_low = True

            spatial_score = 100 if hit_high else (75 if hit_med else (50 if hit_low else c["metrics"]["spatial_match_pct"]))

            candidates.append({
                "mmsi": str(c["mmsi"]),
                "name": c["name"],
                "imo": c.get("imo", "N/A"),
                "flag": c.get("flag", "International"),
                "ship_type": c.get("type", "Commercial Vessel"),
                "overall_score": c["attribution_score"] * 100.0 if c["attribution_score"] <= 1.0 else c["attribution_score"],
                "spatial_match": spatial_score,
                "cpa_nm": c.get("cpa_nm", 0.0),
                "time_delta_hours": c.get("time_delta_hours", 0.0),
                "max_ais_gap_hours": c.get("ais_gap_hours", 0.0),
                "dark_vessel_suspect": c.get("dark_vessel_suspect", False),
                "risk": c.get("risk", "LOW"),
                "trajectory": c.get("trajectory", [])
            })

        # Sort descending by overall attribution score
        candidates.sort(key=lambda x: x["overall_score"], reverse=True)

        return {
            "status": "success",
            "incident_id": incident_id,
            "origin_coordinates": {"lat": origin_lat, "lon": origin_lon},
            "candidate_count": len(candidates),
            "candidates": candidates
        }

    except Exception as e:
        logger.error(f"[correlate_ais_vessels] Correlation error for {incident_id}: {e}")
        return {"status": "failed", "incident_id": incident_id, "error": str(e)}
