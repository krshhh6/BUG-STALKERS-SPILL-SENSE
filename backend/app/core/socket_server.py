import socketio
import os

# We use the AsyncRedisManager so that Celery worker processes 
# can emit WebSocket messages to the FastAPI server process via Redis.
# Fallback to in-memory broker if Redis is not running or localhost is unreachable
redis_url = os.getenv("REDIS_URL", "")

if redis_url and not ("localhost" in redis_url or "127.0.0.1" in redis_url):
    try:
        mgr = socketio.AsyncRedisManager(redis_url)
        sio = socketio.AsyncServer(async_mode='asgi', client_manager=mgr, cors_allowed_origins=[])
    except Exception as e:
        print(f"[Socket.IO] Redis manager init failed ({e}), using in-memory broker")
        sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins=[])
else:
    sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins=[])

# This is the ASGI app that FastAPI will mount
socket_app = socketio.ASGIApp(sio)

@sio.event
async def connect(sid, environ):
    print(f"[Socket.IO] Client connected: {sid}")
    await sio.emit("system_status", {"message": "Connected to SpillSense C2 API"}, to=sid)

@sio.event
async def disconnect(sid):
    print(f"[Socket.IO] Client disconnected: {sid}")
