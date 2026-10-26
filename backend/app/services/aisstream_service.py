# -*- coding: utf-8 -*-
"""
Spill Sense (SIH26143) — AISStream.io Live Real-Time Global WebSocket Stream
Streams live commercial & tanker transponder packets from wss://stream.aisstream.io/v0/stream.
"""

import os
import json
import asyncio
import logging
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

try:
    import aiohttp
except ImportError:
    aiohttp = None

logger = logging.getLogger(__name__)

# ITU-R M.1371 Ship Type codes
AIS_SHIP_TYPES = {
    80: "Crude Oil Tanker",
    81: "Chemical Tanker",
    82: "Gas / LNG / LPG Carrier",
    83: "Oil Products Tanker",
    84: "Bunkering / Asphalt Tanker",
    70: "Container Cargo Vessel",
    71: "Bulk Carrier / Ore Freighter",
    72: "General Cargo",
    60: "Passenger / Cruise Ship",
    52: "Tug / Offshore Supply Vessel",
    30: "Fishing Trawler",
}

class AISStreamService:
    """
    Background WebSocket service connecting to AISStream.io
    to stream real-time vessel packets for the Indian Ocean and EEZ.
    """
    WS_URL = "wss://stream.aisstream.io/v0/stream"
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(AISStreamService, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, api_key: Optional[str] = None):
        if self._initialized:
            return
        from app.core.config import settings
        self.api_key = api_key or os.getenv("AISSTREAM_API_KEY") or getattr(settings, "AISSTREAM_API_KEY", "") or "a5905a6907b539642a35fb5f80dc5b19e423f4a4" (feat(c2): integrate 3D tactical globe with live AIS tracking, glassmorphism UI, and CDSE setup)
        self._vessels: Dict[str, Dict[str, Any]] = {}
        self._is_running = False
        self._task = None
        self._initialized = True
        try:
            from app.services.ais_history_service import AISTrajectoryBuffer
            self.trajectory_buffer = AISTrajectoryBuffer.get_instance()
        except Exception as e:
            logger.warning(f"[AISStream] Trajectory buffer init error: {e}")
            self.trajectory_buffer = None
        logger.info("AISStreamService initialized.")

    def start_background_stream(self, loop=None):
        """Launches the background WebSocket listener task."""
        if not self.api_key or not aiohttp:
            print("[AISStream] Cannot start: Missing API key or aiohttp", flush=True)
            return
        if self._is_running and self._task and not self._task.done():
            return
        self._is_running = True
        try:
            target_loop = loop or asyncio.get_running_loop()
            self._task = target_loop.create_task(self._stream_loop())
            print("[AISStream] Background streaming task scheduled on running event loop.", flush=True)
        except RuntimeError:
            print("[AISStream] No running event loop detected; launching in dedicated background thread.", flush=True)
            t = threading.Thread(target=self._run_thread, daemon=True)
            t.start()

    def _run_thread(self):
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            new_loop.run_until_complete(self._stream_loop())
        finally:
            new_loop.close()

    async def _stream_loop(self):
        """Continuously maintains WebSocket connection to AISStream.io."""
        subscribe_message = {
            "APIKey": self.api_key,
            # Bounding box strictly covering Indian Waters & EEZ: Arabian Sea, Bay of Bengal, Lakshadweep, Andaman Sea
            "BoundingBoxes": [[[5.0, 65.0], [26.0, 96.0]]],
            "FilterMessageTypes": ["PositionReport", "StandardClassBPositionReport", "ShipStaticData"]
        }

        while self._is_running:
            try:
                print(f"[AISStream] Connecting to {self.WS_URL} for Indian EEZ...", flush=True)
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(self.WS_URL, timeout=15) as ws:
                        await ws.send_str(json.dumps(subscribe_message))
                        print("[AISStream] Connected & Indian EEZ surveillance subscription active.", flush=True)

                        async for msg in ws:
                            if not self._is_running:
                                break
                            raw_text = msg.data.decode('utf-8') if isinstance(msg.data, (bytes, bytearray)) else msg.data
                            if not raw_text:
                                continue
                            try:
                                data = json.loads(raw_text)
                                mtype = data.get("MessageType")
                                if mtype == "SubscriptionConfirmation":
                                    print("[AISStream] Indian EEZ subscription confirmed by server.", flush=True)
                                    continue
                                
                                self._process_packet(data)
                            except Exception as parse_err:
                                logger.debug(f"[AISStream] Parse error: {parse_err}")

            except Exception as e:
                print(f"[AISStream] Stream error: {e}. Reconnecting in 10s...", flush=True)
                await asyncio.sleep(10)

    def _process_packet(self, data: Dict[str, Any]):
        """Parses a real-time AIS packet and updates the vessel cache for Indian waters."""
        meta = data.get("MetaData", {})
        mmsi = str(meta.get("MMSI") or "")
        if not mmsi:
            return

        lat = meta.get("latitude")
        lon = meta.get("longitude")
        if lat is None or lon is None:
            return

        f_lat = float(lat)
        f_lon = float(lon)
        # 1. Geographic restriction: Indian EEZ & territorial waters only (5°N - 26°N, 65°E - 96°E)
        if not (5.0 <= f_lat <= 26.0 and 65.0 <= f_lon <= 96.0):
            return

        raw_name = meta.get("ShipName", "").strip()
        name_upper = raw_name.upper()

        # 2. Strict Polluter Filter: Discard yachts, sailing craft, and small pleasure boats
        if name_upper.startswith("S/Y") or name_upper.startswith("M/Y") or "YACHT" in name_upper or "SAIL" in name_upper:
            return

        message = data.get("Message", {})
        pos_report = message.get("PositionReport") or message.get("StandardClassBPositionReport") or {}
        static_data = message.get("ShipStaticData") or {}

        existing = self._vessels.get(mmsi, {})

        raw_name = meta.get("ShipName", "").strip() or static_data.get("Name", "").strip()
        name = raw_name if raw_name else existing.get("name", f"MMSI-{mmsi}")
        
        if "Sog" in pos_report:
            sog = float(pos_report["Sog"])
        else:
            sog = float(existing.get("sog", 0.0))

        if "Cog" in pos_report:
            cog = float(pos_report["Cog"])
        else:
            cog = float(existing.get("cog", 0.0))
        
        type_code = static_data.get("Type")
        if type_code is not None:
            v_type = AIS_SHIP_TYPES.get(int(type_code), "Cargo / Commercial Vessel")
        else:
            v_type = existing.get("type", "Cargo / Commercial Vessel")

        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S GMT")

        is_tanker = any(t in v_type.lower() for t in ["tanker", "crude", "oil", "gas", "lng", "lpg", "chemical", "bunker"])
        nav_status = f"Underway ({sog:.1f} kn)" if sog >= 0.5 else "Moored / At Anchor"

        # Persist to rolling trajectory buffer with kinematic downsampling
        if getattr(self, "trajectory_buffer", None):
            try:
                self.trajectory_buffer.ingest_packet(
                    mmsi=mmsi,
                    name=name,
                    imo=str(meta.get("IMO") or static_data.get("ImoNumber") or existing.get("imo") or "N/A"),
                    flag="International",
                    ship_type=v_type,
                    is_tanker=is_tanker,
                    lat=f_lat,
                    lon=f_lon,
                    sog=sog,
                    cog=cog
                )
            except Exception as e:
                logger.debug(f"[AISStream] Buffer ingest error: {e}")

        is_new = mmsi not in self._vessels
        self._vessels[mmsi] = {
            "mmsi": mmsi,
            "imo": str(meta.get("IMO") or static_data.get("ImoNumber") or existing.get("imo") or "N/A"),
            "name": name,
            "flag": "International",
            "type": v_type,
            "lat": float(lat),
            "lng": float(lon),
            "sog": sog,
            "cog": cog,
            "nav_status": nav_status,
            "is_tanker": is_tanker,
            "risk": "HIGH" if (is_tanker and 0.5 <= sog <= 6.0) else ("MEDIUM" if is_tanker else "LOW"),
            "last_ping_utc": meta.get("time_utc", now_utc),
            "_updated_timestamp": datetime.now(timezone.utc).timestamp()
        }
        if is_new and (len(self._vessels) <= 5 or len(self._vessels) % 50 == 0):
            print(f"[AISStream] Real-time tracked: {len(self._vessels)} vessels. Latest: {name} ({v_type}) at [{lat:.3f}, {lon:.3f}]", flush=True)

    def get_live_vessels(
        self,
        latmin: float,
        latmax: float,
        lonmin: float,
        lonmax: float
    ) -> List[Dict[str, Any]]:
        """Filters cached live vessels for a given incident bounding box."""
        center_lat = (latmin + latmax) / 2.0
        center_lon = (lonmin + lonmax) / 2.0
        now = datetime.now(timezone.utc).timestamp()

        results = []
        for mmsi, v in list(self._vessels.items()):
            # Prune if not updated in 45 minutes
            if now - v.get("_updated_timestamp", now) > 2700:
                self._vessels.pop(mmsi, None)
                continue

            v_lat = v["lat"]
            v_lon = v["lng"]
            if latmin <= v_lat <= latmax and lonmin <= v_lon <= lonmax:
                deg_dist = ((v_lat - center_lat)**2 + (v_lon - center_lon)**2)**0.5
                cpa = round(deg_dist * 60.0, 1)
                
                v_copy = dict(v)
                v_copy["cpa_nm"] = cpa
                v_copy.pop("_updated_timestamp", None)
                results.append(v_copy)

        return results

    def get_all_vessels(self, limit: int = 150) -> List[Dict[str, Any]]:
        """Returns currently tracked live vessels sorted by recency."""
        v_list = list(self._vessels.values())
        v_list.sort(key=lambda x: x.get("_updated_timestamp", 0), reverse=True)
        res = []
        for v in v_list[:limit]:
            vc = dict(v)
            vc.pop("_updated_timestamp", None)
            res.append(vc)
        return res

