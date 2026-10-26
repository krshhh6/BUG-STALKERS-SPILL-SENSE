from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    description="Automated Oil Spill Detection, Backward Lagrangian Drift & AIS Attribution C2 API"
)

# CORS Policy Configuration (tech_stack.md §11)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_private_network_access_header(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


@app.get("/")
async def root():
    return {
        "status": "ONLINE",
        "service": settings.PROJECT_NAME,
        "compliance": ["MARPOL Annex I", "Indian Merchant Shipping Act 1958 §356"],
        "docs": "/docs"
    }

@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "HEALTHY",
        "database": "PostgreSQL 16 + PostGIS 3.4",
        "task_queue": "Celery + Redis",
        "sar_source": "Copernicus Data Space Ecosystem (CDSE)"
    }


from app.api.v1.endpoints import router as api_router
app.include_router(api_router, prefix=settings.API_V1_STR)

from app.api.v1.sentinelhub_router import router as sh_router
app.include_router(sh_router, prefix=settings.API_V1_STR)

from app.api.v1.detect_router import router as detect_router
app.include_router(detect_router, prefix=settings.API_V1_STR)
app.include_router(detect_router, prefix="/api")

from app.api.v1.ml_metrics_router import router as ml_router
app.include_router(ml_router, prefix=settings.API_V1_STR)
app.include_router(ml_router, prefix="/api")

from app.api.v1.ais_router import router as ais_router
app.include_router(ais_router, prefix=settings.API_V1_STR)
app.include_router(ais_router, prefix="/api")

from app.api.v1.active_learning_router import router as active_learning_router
app.include_router(active_learning_router, prefix=f"{settings.API_V1_STR}/active-learning")
app.include_router(active_learning_router, prefix="/api/active-learning")

from app.api.v1.setup_router import router as setup_router
app.include_router(setup_router, prefix=settings.API_V1_STR)
app.include_router(setup_router, prefix="/api")


# Mount Static Files for Forensic Evidence PDF Downloads
import os
from fastapi.staticfiles import StaticFiles

evidence_storage_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "storage", "evidence")
os.makedirs(evidence_storage_dir, exist_ok=True)
app.mount("/static/evidence", StaticFiles(directory=evidence_storage_dir), name="evidence_static")

# Mount Socket.IO for Real-Time Event Broadcasting
from app.core.socket_server import socket_app
app.mount("/", socket_app)

@app.on_event("startup")
async def startup_event():
    from app.services.aisstream_service import AISStreamService
    stream = AISStreamService()
    stream.start_background_stream()
