# -*- coding: utf-8 -*-
"""
Spill Sense (SIH26143) — C2 System & API Key Setup Router
Provides secure endpoints to view configured API key statuses, live-save keys to .env,
and run active connectivity diagnostics (AISStream WebSocket & Cesium Ion).
"""

import os
import re
import json
import asyncio
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/setup", tags=["C2 API Key Setup & Feeds"])

# Canonical known environment variables for Spill Sense C2
KNOWN_KEYS = {
    "CESIUM_ION_TOKEN": {
        "title": "Cesium Ion Access Token",
        "service": "Cesium Ion 3D Geospatial Cloud",
        "description": "Unlocks high-resolution Cesium World Terrain, bathymetry, photorealistic 3D tiles, and satellite basemaps.",
        "placeholder": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
        "docsUrl": "https://ion.cesium.com/tokens",
        "required": False,
    },
    "AISSTREAM_API_KEY": {
        "title": "AISStream.io Live WebSocket Key",
        "service": "AISStream.io Global Maritime Feed",
        "description": "Powers real-time transponder telemetry for commercial tankers and cargo vessels across the Indian EEZ.",
        "placeholder": "a5905a6907b539642a35fb5f80dc5b19e423f4a4",
        "docsUrl": "https://aisstream.io",
        "required": True,
    },
    "CDSE_CLIENT_ID": {
        "title": "Copernicus CDSE Client ID",
        "service": "Copernicus Data Space Ecosystem",
        "description": "Accesses Sentinel-1 C-SAR IW GRD satellite passes and raw radar imagery for oil slick delineation.",
        "placeholder": "cdse-oauth-client-id",
        "docsUrl": "https://dataspace.copernicus.eu",
        "required": False,
    },
    "CDSE_CLIENT_SECRET": {
        "title": "Copernicus CDSE Client Secret",
        "service": "Copernicus Data Space Ecosystem",
        "description": "OAuth2 client secret for automated Sentinel-1 product catalog queries and download pipelines.",
        "placeholder": "••••••••••••••••",
        "docsUrl": "https://dataspace.copernicus.eu",
        "required": False,
    },
    "AISHUB_USERNAME": {
        "title": "AISHub Service Username",
        "service": "AISHub Maritime Data Network",
        "description": "Secondary terrestrial AIS stream provider for vessel identity reconciliation.",
        "placeholder": "aishub_account_user",
        "docsUrl": "https://www.aishub.net",
        "required": False,
    },
}

class KeyUpdatePayload(BaseModel):
    keys: Dict[str, str] = Field(..., description="Map of environment variable names to their new secret values")

def mask_secret(value: Optional[str], key_name: str = "") -> Optional[str]:
    """Masks secret values showing only prefix and suffix for security, preserves usernames."""
    if not value or not isinstance(value, str):
        return None
    trimmed = value.strip()
    if "USERNAME" in key_name.upper() or "USER" in key_name.upper():
        return trimmed
    if len(trimmed) <= 8:
        return "••••••••"
    if len(trimmed) <= 16:
        return f"{trimmed[:3]}••••{trimmed[-3:]}"
    return f"{trimmed[:6]}••••••••{trimmed[-4:]}"

def get_env_file_path() -> Path:
    """Resolves path to the backend .env file."""
    # Current file is backend/app/api/v1/setup_router.py -> backend/.env
    backend_root = Path(__file__).resolve().parent.parent.parent.parent
    return backend_root / ".env"

def get_frontend_env_file_path() -> Path:
    """Resolves path to the frontend .env.local file."""
    repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    return repo_root / "frontend" / ".env.local"

def update_dotenv_file(filepath: Path, updates: Dict[str, str]) -> None:
    """Updates or appends key-value pairs in a .env file cleanly preserving comments."""
    lines = []
    if filepath.exists():
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            logger.warning(f"Could not read {filepath}: {e}")

    existing_keys = set()
    new_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            k, _ = stripped.split("=", 1)
            k = k.strip()
            existing_keys.add(k)
            if k in updates:
                new_val = updates[k].strip()
                new_lines.append(f'{k}="{new_val}"\n' if " " in new_val else f"{k}={new_val}\n")
                continue
        new_lines.append(line)

    # Append any keys that weren't present
    for k, v in updates.items():
        if k not in existing_keys and v.strip():
            val = v.strip()
            new_lines.append(f'{k}="{val}"\n' if " " in val else f"{k}={val}\n")

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


@router.get("/status")
async def get_key_setup_status():
    """
    Returns the configuration status and masked previews of all known API keys.
    Does NOT leak full private keys over the wire.
    """
    statuses = {}
    for key_name, meta in KNOWN_KEYS.items():
        val = os.getenv(key_name, "").strip()
        is_set = bool(val)
        statuses[key_name] = {
            "key": key_name,
            "title": meta["title"],
            "service": meta["service"],
            "description": meta["description"],
            "placeholder": meta["placeholder"],
            "docsUrl": meta["docsUrl"],
            "required": meta["required"],
            "configured": is_set,
            "maskedValue": mask_secret(val, key_name) if is_set else None,
            "length": len(val) if is_set else 0,
        }

    return {
        "status": "OK",
        "keys": statuses,
        "configuredCount": sum(1 for v in statuses.values() if v["configured"]),
        "totalKnown": len(statuses),
    }


@router.post("/keys")
async def save_keys(payload: KeyUpdatePayload, background_tasks: BackgroundTasks):
    """
    Saves updated API keys into backend/.env and frontend/.env.local,
    updates live os.environ, and re-arms live background services (AISStream).
    """
    updates = payload.keys
    sanitized: Dict[str, str] = {}
    for k, v in updates.items():
        if k in KNOWN_KEYS:
            sanitized[k] = v.strip() if v else ""

    if not sanitized:
        raise HTTPException(status_code=400, detail="No recognized API key environment variables provided.")

    # 1. Update live process.env
    for k, v in sanitized.items():
        if v:
            os.environ[k] = v
        elif k in os.environ:
            del os.environ[k]

    # 2. Persist to backend .env
    backend_env = get_env_file_path()
    try:
        update_dotenv_file(backend_env, sanitized)
    except Exception as e:
        logger.error(f"Failed writing backend .env: {e}")
        raise HTTPException(status_code=500, detail=f"Failed writing .env file: {str(e)}")

    # 3. Mirror frontend-relevant keys to frontend/.env.local (e.g. VITE_CESIUM_ION_TOKEN)
    frontend_updates = {}
    if "CESIUM_ION_TOKEN" in sanitized:
        frontend_updates["VITE_CESIUM_ION_TOKEN"] = sanitized["CESIUM_ION_TOKEN"]
    if "AISSTREAM_API_KEY" in sanitized:
        frontend_updates["VITE_AISSTREAM_API_KEY"] = sanitized["AISSTREAM_API_KEY"]

    if frontend_updates:
        frontend_env = get_frontend_env_file_path()
        try:
            update_dotenv_file(frontend_env, frontend_updates)
        except Exception as e:
            logger.warning(f"Could not mirror to frontend .env.local: {e}")

    # 4. If AISSTREAM_API_KEY changed, re-arm the AISStreamService
    if "AISSTREAM_API_KEY" in sanitized and sanitized["AISSTREAM_API_KEY"]:
        try:
            from app.services.aisstream_service import AISStreamService
            stream = AISStreamService()
            stream.api_key = sanitized["AISSTREAM_API_KEY"]
            stream.start_background_stream()
            print(f"[KeySetup] Re-armed AISStreamService with new key: {mask_secret(stream.api_key)}", flush=True)
        except Exception as err:
            logger.error(f"Error re-arming AISStreamService: {err}")

    return {
        "status": "SUCCESS",
        "message": f"Successfully persisted {len(sanitized)} key(s) to environment.",
        "updatedKeys": list(sanitized.keys()),
    }


@router.post("/test-ais")
async def test_aisstream_key(key_override: Optional[str] = None):
    """
    Directly tests WebSocket connectivity against wss://stream.aisstream.io/v0/stream
    with the current or newly supplied API key.
    """
    key = key_override or os.getenv("AISSTREAM_API_KEY", "")
    if not key:
        return {
            "success": False,
            "error": "No AISStream API key configured to test.",
            "latencyMs": 0,
        }

    try:
        import aiohttp
        import time
        start_t = time.perf_counter()
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect("wss://stream.aisstream.io/v0/stream", timeout=8) as ws:
                subscribe_msg = {
                    "APIKey": key,
                    "BoundingBoxes": [[[5.0, 65.0], [26.0, 96.0]]],
                    "FilterMessageTypes": ["PositionReport"],
                }
                await ws.send_str(json.dumps(subscribe_msg))
                raw = await asyncio.wait_for(ws.receive(), timeout=6)
                elapsed = round((time.perf_counter() - start_t) * 1000, 1)

                data = raw.data.decode("utf-8") if isinstance(raw.data, (bytes, bytearray)) else str(raw.data)
                parsed = json.loads(data) if data else {}

                msg_type = parsed.get("MessageType")
                if msg_type == "SubscriptionConfirmation":
                    return {
                        "success": True,
                        "latencyMs": elapsed,
                        "message": "Subscription confirmed. Feed active for Indian Ocean & EEZ surveillance.",
                        "maskedKey": mask_secret(key),
                    }
                elif "error" in data.lower() or "unauthor" in data.lower():
                    return {
                        "success": False,
                        "latencyMs": elapsed,
                        "error": data,
                        "maskedKey": mask_secret(key),
                    }
                else:
                    return {
                        "success": True,
                        "latencyMs": elapsed,
                        "message": f"Connected. Received initial packet type: {msg_type or 'Stream Ready'}",
                        "maskedKey": mask_secret(key),
                    }
    except Exception as e:
        return {
            "success": False,
            "error": f"Connection test failed: {str(e)}",
            "latencyMs": 0,
        }


@router.post("/test-cesium")
async def test_cesium_token(token_override: Optional[str] = None):
    """
    Tests validity of Cesium Ion Access Token against the Cesium Ion Asset REST API.
    """
    token = token_override or os.getenv("CESIUM_ION_TOKEN", "")
    if not token:
        return {
            "success": False,
            "error": "No Cesium Ion Token configured.",
        }

    try:
        import aiohttp
        import time
        start_t = time.perf_counter()
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {token}"}
            async with session.get("https://api.cesium.com/v1/assets/1", headers=headers, timeout=8) as resp:
                elapsed = round((time.perf_counter() - start_t) * 1000, 1)
                if resp.status == 200:
                    asset_info = await resp.json()
                    return {
                        "success": True,
                        "latencyMs": elapsed,
                        "assetName": asset_info.get("name", "Cesium World Terrain"),
                        "message": "Cesium Ion Token verified. World Terrain & Photorealistic Assets unlocked.",
                        "maskedToken": mask_secret(token),
                    }
                else:
                    text = await resp.text()
                    return {
                        "success": False,
                        "latencyMs": elapsed,
                        "error": f"Cesium Ion rejected token (HTTP {resp.status}): {text}",
                        "maskedToken": mask_secret(token),
                    }
    except Exception as e:
        return {
            "success": False,
            "error": f"Cesium test error: {str(e)}",
        }
