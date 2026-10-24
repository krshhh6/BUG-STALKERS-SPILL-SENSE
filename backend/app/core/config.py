from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    PROJECT_NAME: str = "Spill Sense C2 Platform (SIH26143)"
    API_V1_STR: str = "/api/v1"
    
    # Database (PostgreSQL + PostGIS)
    DATABASE_URL: str = "postgresql+asyncpg://spill_user:spill_pass_26143@localhost:5432/spill_sense"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Copernicus CDSE Credentials
    CDSE_USERNAME: str = ""
    CDSE_PASSWORD: str = ""
    
    # AISStream API Key
    AISSTREAM_API_KEY: str = ""

    # CORS Origins
    CORS_ORIGINS: List[str] = ["http://localhost:8080", "http://localhost:3000", "http://localhost:5173", "*"]

    class Config:
        case_sensitive = True
        env_file = ".env"
        extra = "ignore"

settings = Settings()
