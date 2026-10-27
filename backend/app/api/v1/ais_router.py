# -*- coding: utf-8 -*-
"""
Spill Sense (SIH26143) — AISHub Live Maritime Routing & Attribution Endpoints
"""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Query
from pydantic import BaseModel
from app.services.aishub_service import AISHubService
from app.services.aisstream_service import AISStreamService
from datetime import datetime, timezone

router = APIRouter(prefix="/ais", tags=["AIS Maritime Surveillance"])

ais_service = AISHubService()
stream_service = AISStreamService()

class CorrelationWeights(BaseModel):
    dist: float = 0.30
    time: float = 0.25
    gap: float = 0.20
    type: float = 0.15

class CorrelateRequest(BaseModel):
    incident_id: str = "INC-2026-001"
    lat: float = 18.743
    lng: float = 71.218
    origin_lat: Optional[float] = None
    origin_lng: Optional[float] = None
    horizon_hours: Optional[float] = 24.0
    use_historical: bool = True
    radius_nm: float = 30.0
    username: Optional[str] = None
    weights: CorrelationWeights = CorrelationWeights()

@router.get("/live")
async def get_live_ais_vessels(
    latmin: float = Query(17.5, description="South latitude"),
    latmax: float = Query(19.5, description="North latitude"),
    lonmin: float = Query(70.0, description="West longitude"),
    lonmax: float = Query(72.5, description="East longitude"),
    username: Optional[str] = Query(None, description="Optional AISHub username")
):
    """
    Fetches real-time AIS vessel locations from live AISStream.io feed or AISHub webservice.
    """
    # 1. Check if AISStream WebSocket has live vessels in this bounding box
    stream_vessels = stream_service.get_live_vessels(latmin, latmax, lonmin, lonmax)
    if stream_vessels:
        return {
            "source": "AISSTREAM_LIVE_WEBSOCKET",
            "count": len(stream_vessels),
            "vessels": stream_vessels,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    # 2. Otherwise query AISHub or fall back to calibrated Indian EEZ dataset
    return ais_service.get_live_vessels(latmin, latmax, lonmin, lonmax, username=username)

@router.get("/stream-status")
async def get_stream_status():
    """Returns the live status of the AISStream WebSocket connection and active vessels."""
    all_vessels = stream_service.get_all_vessels()
    return {
        "status": "ONLINE" if stream_service._is_running else "OFFLINE",
        "service": "AISStream.io Live Global WebSocket",
        "api_key_configured": bool(stream_service.api_key),
        "total_live_vessels_tracked": len(all_vessels),
        "vessels": all_vessels
    }


@router.get("/historical-candidates")
async def get_historical_candidates(
    incident_id: str = Query("INC-2026-001"),
    origin_lat: float = Query(18.663),
    origin_lng: float = Query(71.158),
    horizon_hours: float = Query(24.0, description="Hours backwards from now for discharge window"),
    radius_nm: float = Query(35.0, description="Spatial search radius in nautical miles"),
    w_dist: float = Query(0.30),
    w_time: float = Query(0.25),
    w_gap: float = Query(0.25),
    w_type: float = Query(0.20)
):
    """
    Queries the 14-day rolling historical trajectory buffer to extract vessels whose
    trajectories intersected the reconstructed backward Lagrangian drift origin envelope.
    """
    from app.services.ais_history_service import AISTrajectoryBuffer
    buf = AISTrajectoryBuffer.get_instance()
    now_ts = datetime.now(timezone.utc).timestamp()
    t_discharge = now_ts - (horizon_hours * 3600.0)

    weights = {"dist": w_dist, "time": w_time, "gap": w_gap, "type": w_type}
    candidates = buf.query_historical_candidates(
        origin_lat=origin_lat,
        origin_lon=origin_lng,
        start_time=t_discharge - 7200.0,
        end_time=t_discharge + 7200.0,
        radius_nm=radius_nm,
        weights=weights
    )

    return {
        "status": "SUCCESS",
        "incident_id": incident_id,
        "feed_source": "HISTORICAL_TRAJECTORY_BUFFER_4D",
        "discharge_origin": {"lat": origin_lat, "lon": origin_lng},
        "horizon_hours": horizon_hours,
        "total_vessels_screened": len(candidates),
        "candidates": candidates
    }


@router.post("/score")
@router.post("/attribute")
async def score_vessels(req: CorrelateRequest):
    """
    Computes explainable vessel attribution scores across live/cached AISHub vessels
    using customizable spatiotemporal sensitivity weights.
    """
    deg_delta = req.radius_nm / 60.0
    latmin = req.lat - deg_delta
    latmax = req.lat + deg_delta
    lonmin = req.lng - deg_delta
    lonmax = req.lng + deg_delta

    # If historical mode is enabled, query the rolling trajectory buffer at the discharge origin!
    if req.use_historical and req.horizon_hours:
        from app.services.ais_history_service import AISTrajectoryBuffer
        buf = AISTrajectoryBuffer.get_instance()
        now_ts = datetime.now(timezone.utc).timestamp()
        t_discharge = now_ts - (req.horizon_hours * 3600.0)
        target_lat = req.origin_lat if req.origin_lat is not None else req.lat
        target_lng = req.origin_lng if req.origin_lng is not None else req.lng

        hist_candidates = buf.query_historical_candidates(
            origin_lat=target_lat,
            origin_lon=target_lng,
            start_time=t_discharge - 7200.0,
            end_time=t_discharge + 7200.0,
            radius_nm=req.radius_nm,
            weights=req.weights.dict()
        )
        if hist_candidates:
            return {
                "status": "SUCCESS",
                "incident_id": req.incident_id,
                "feed_source": "HISTORICAL_TRAJECTORY_BUFFER_4D",
                "total_vessels_screened": len(hist_candidates),
                "candidates": hist_candidates,
                "active_weights": req.weights.dict()
            }

    stream_vessels = stream_service.get_live_vessels(latmin, latmax, lonmin, lonmax)
    if stream_vessels:
        raw_vessels = stream_vessels
        feed_source = "AISSTREAM_LIVE_WEBSOCKET"
    else:
        feed = ais_service.get_live_vessels(latmin, latmax, lonmin, lonmax, username=req.username)
        raw_vessels = feed.get("vessels", [])
        feed_source = feed.get("source", "AISHUB_API")

    w = req.weights
    total_w = w.dist + w.time + w.gap + w.type or 1.0

    ranked = []
    for v in raw_vessels:
        # Distance score: closer CPA => higher score
        cpa = v.get("cpa_nm", 10.0)
        s_dist = max(0.0, min(1.0, 1.0 - (cpa / req.radius_nm)))

        # AIS Gap score: larger gap near slick => higher suspicion
        gap_h = v.get("ais_gap_hours", 0.0)
        s_gap = min(1.0, gap_h / 4.0)

        # Vessel type score
        v_type = v.get("type", "").lower()
        if "crude" in v_type:
            s_type = 0.95
        elif "product" in v_type or "bunker" in v_type:
            s_type = 0.85
        elif "chemical" in v_type:
            s_type = 0.70
        elif "cargo" in v_type or "container" in v_type:
            s_type = 0.35
        else:
            s_type = 0.20

        # Time score (derived or default 0.80)
        s_time = v.get("timeScore", 0.80)

        # Weighted Attribution Score
        score = (w.dist * s_dist + w.time * s_time + w.gap * s_gap + w.type * s_type) / total_w
        score = round(min(1.0, max(0.05, score)), 2)

        v_copy = dict(v)
        v_copy["attribution_score"] = score
        v_copy["metrics"] = {
            "spatial_match_pct": round(s_dist * 100),
            "temporal_alignment_pct": round(s_time * 100),
            "dark_gap_suspicion_pct": round(s_gap * 100),
            "vessel_risk_prior_pct": round(s_type * 100)
        }
        ranked.append(v_copy)

    # Sort descending by attribution score
    ranked.sort(key=lambda x: x["attribution_score"], reverse=True)

    # Assign ranks
    for idx, r in enumerate(ranked):
        r["rank"] = idx + 1

    return {
        "status": "SUCCESS",
        "incident_id": req.incident_id,
        "feed_source": feed_source,
        "total_vessels_screened": len(ranked),
        "candidates": ranked,
        "active_weights": w.dict()
    }
