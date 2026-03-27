"""Application configuration loaded from environment variables."""

from __future__ import annotations

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central settings object – values are read from env vars / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ── Database ────────────────────────────────────────────────────────
    DATABASE_URL: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/sloth"
    )

    @field_validator("DATABASE_URL", mode="after")
    @classmethod
    def _fix_db_scheme(cls, v: str) -> str:
        """Railway gives ``postgresql://`` – swap to ``postgresql+asyncpg://``."""
        if v.startswith("postgresql://"):
            v = v.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif v.startswith("postgres://"):
            v = v.replace("postgres://", "postgresql+asyncpg://", 1)
        return v

    # ── Redis ───────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379"

    # ── AI / LLM API keys ──────────────────────────────────────────────
    FAL_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    GOOGLE_CREDENTIALS_JSON: str = ""
    VEO_PROJECT_ID: str = ""
    VEO_ACCESS_TOKEN: str = ""
    VEO_GCS_BUCKET: str = ""
    VEO_LOCATION: str = "us-central1"
    VEO_MODEL_ID: str = "veo-3.1-generate-preview"
    TTS_VOICE_NAME: str = "en-US-Studio-O"
    TTS_PITCH: float = 0.0
    TTS_EFFECTS_PROFILE_ID: str = "headphone-class-device"
    TTS_MAX_SCRIPT_CHARS: int = 50000

    # ── Scraping ────────────────────────────────────────────────────────
    FIRECRAWL_API_KEY: str = ""

    # ── Zernio (ad placement / analytics) ──────────────────────────────
    ZERNIO_API_KEY: str = ""
    LATE_API_KEY: str = ""
    LATE_API_BASE_URL: str = "https://getlate.dev/api/v1"
    MUBERT_COMPANY_ID: str = ""
    MUBERT_LICENSE_TOKEN: str = ""

    # ── Cloudflare R2 (S3-compatible storage) ──────────────────────────
    R2_ENDPOINT: str = ""
    R2_ACCESS_KEY: str = ""
    R2_SECRET_KEY: str = ""
    R2_BUCKET: str = "sloth-assets"

    # ── CORS ────────────────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:3002"]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def _parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        """Accept a comma-separated string **or** a JSON list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v


# Singleton – import this wherever you need config values.
settings = Settings()
