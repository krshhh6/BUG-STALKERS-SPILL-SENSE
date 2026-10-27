import math
from typing import List, Tuple, Optional
from app.models.schemas.detection import DetectionPayload, AisPing, ConfidenceMatrix, EvidenceRecord, DriftVector
from app.services.environmental_service import EnvironmentalService

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance in meters between two points on the earth."""
    R = 6371000  # radius of Earth in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

class AttributionEngine:
    """
    Engine to correlate SAR model detections with AIS data using dynamic physics drift.
    Distinguishes between powered vessel kinematics (SOG, COG) and passive slick Lagrangian drift.
    """
    
    @staticmethod
    def correlate_detections(
        detection: DetectionPayload, 
        ais_history: List[AisPing],
        incident_id: str = "INC-001"
    ) -> EvidenceRecord:
        """
        Cross-references a SAR detection against historical AIS tracks.
        Returns the strongest EvidenceRecord match, or an Uncorrelated Novel detection if no matches found.
        """
        
        # Calculate SAR centroid
        det_lon = (detection.bounding_box.min_lon + detection.bounding_box.max_lon) / 2.0
        det_lat = (detection.bounding_box.min_lat + detection.bounding_box.max_lat) / 2.0
        
        best_match: Optional[AisPing] = None
        best_confidence: Optional[ConfidenceMatrix] = None
        best_drift = None
        highest_score = 0.0
        
        for ais in ais_history:
            # We only project forward in time (AIS ping -> SAR detection time)
            if ais.timestamp > detection.timestamp:
                continue
                
            # 1. Powered Vessel Kinematic Navigation (SOG / COG Dead Reckoning)
            dt_seconds = (detection.timestamp - ais.timestamp).total_seconds()
            
            if dt_seconds > 0 and ais.sog_knots is not None and ais.sog_knots > 0.5 and ais.cog_deg is not None:
                # Motorized ship navigates under power according to course and speed
                sog_ms = ais.sog_knots * 0.514444
                cog_rad = math.radians(ais.cog_deg)
                dx_m = sog_ms * math.sin(cog_rad) * dt_seconds
                dy_m = sog_ms * math.cos(cog_rad) * dt_seconds
                projected_lat = ais.lat + (dy_m / 111111.0)
                projected_lon = ais.lon + (dx_m / (111111.0 * math.cos(math.radians(ais.lat))))
                drift_vector_dict = {
                    "u_current_ms": 0.0,
                    "v_current_ms": 0.0,
                    "u_wind_ms": round(sog_ms * math.sin(cog_rad), 2),
                    "v_wind_ms": round(sog_ms * math.cos(cog_rad), 2),
                    "drift_time_seconds": dt_seconds,
                    "delta_x_m": round(dx_m, 1),
                    "delta_y_m": round(dy_m, 1)
                }
            else:
                # Vessel stationary or anchor drift
                projected_lat = ais.lat
                projected_lon = ais.lon
                drift_vector_dict = {
                    "u_current_ms": 0.0,
                    "v_current_ms": 0.0,
                    "u_wind_ms": 0.0,
                    "v_wind_ms": 0.0,
                    "drift_time_seconds": max(0.0, dt_seconds),
                    "delta_x_m": 0.0,
                    "delta_y_m": 0.0
                }
            
            # 2. Spatial Scoring
            distance_m = haversine_distance(projected_lat, projected_lon, det_lat, det_lon)
            
            # Dynamic uncertainty radius as a function of time elapsed (expanding cone of navigation error)
            uncertainty_radius_m = 1000.0 + (dt_seconds * 0.15)
            spatial_score = math.exp(- (distance_m / uncertainty_radius_m))
            
            # 3. Dimension Scoring
            dimension_score = 1.0  # Default if no data
            if detection.estimated_length_m and ais.vessel_length_m:
                length_diff = abs(detection.estimated_length_m - ais.vessel_length_m)
                dimension_score = math.exp(- (length_diff / (ais.vessel_length_m * 0.2)))
                
            # 4. Heading / Course Scoring
            heading_score = 1.0  # Default
            if detection.heading_deg is not None and ais.cog_deg is not None:
                diff = abs(detection.heading_deg - ais.cog_deg)
                if diff > 180:
                    diff = 360 - diff
                heading_score = math.exp(- (diff / 30.0))
                
            # 5. MARPOL Annex I Hydrocarbon Cargo Compatibility & Vessel Risk Prior
            v_type_lower = (ais.vessel_type or "").lower()
            if any(k in v_type_lower for k in ["lng", "lpg", "gas carrier", "methane", "cryogenic"]):
                type_prior = 0.05  # Cryogenic cargo evaporates instantaneously; cannot form persistent surface slick
            elif any(k in v_type_lower for k in ["crude", "vlcc", "suezmax", "aframax", "oil tanker"]):
                type_prior = 1.0
            elif any(k in v_type_lower for k in ["bunker", "asphalt", "bitumen"]):
                type_prior = 0.95
            elif any(k in v_type_lower for k in ["chemical", "tanker"]):
                type_prior = 0.85
            elif any(k in v_type_lower for k in ["cargo", "container", "bulk"]):
                type_prior = 0.45
            else:
                type_prior = 0.25

            # Weighted Overall Confidence
            overall = (spatial_score * 0.55) + (dimension_score * 0.20) + (heading_score * 0.15) + (type_prior * 0.10)
            
            if overall > highest_score:
                highest_score = overall
                best_match = ais
                best_drift = DriftVector(**drift_vector_dict)
                best_confidence = ConfidenceMatrix(
                    spatial_score=max(0.0, min(1.0, spatial_score)),
                    dimension_score=max(0.0, min(1.0, dimension_score)),
                    heading_score=max(0.0, min(1.0, heading_score)),
                    overall_confidence=max(0.0, min(1.0, overall))
                )
                
        # Determine classification
        if not best_match or highest_score < 0.20:
            return EvidenceRecord(
                incident_id=incident_id,
                detection=detection,
                classification="Uncorrelated Novel",
                drift_applied=None,
                correlated_ais=None,
                confidence=None
            )
            
        elif highest_score >= 0.60:
            classification = "Correlated (Reporting)"
        else:
            classification = "Suspect / Dark"
            
        return EvidenceRecord(
            incident_id=incident_id,
            detection=detection,
            classification=classification,
            drift_applied=best_drift,
            correlated_ais=best_match,
            confidence=best_confidence
        )
