from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SQLALCHEMY_DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://postgres:postgres@localhost:5432/spillsense"
)
if SQLALCHEMY_DATABASE_URL and SQLALCHEMY_DATABASE_URL.startswith("postgres://"):
    SQLALCHEMY_DATABASE_URL = SQLALCHEMY_DATABASE_URL.replace("postgres://", "postgresql://", 1)

use_sqlite = False
try:
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        connect_args={"connect_timeout": 2} if "postgres" in SQLALCHEMY_DATABASE_URL else {}
    )
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
except Exception:
    use_sqlite = True

if use_sqlite:
    # Resilient offline persistence on Drive D: without Docker
    engine = create_engine(
        "sqlite:///D:/SIH-26143-OIL-Spill/SIH-26143-OIL-Spill/spill_sense.db",
        connect_args={"check_same_thread": False}
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Dependency to generate DB sessions for FastAPI routes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
