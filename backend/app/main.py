"""FastAPI application entry-point for the Sloth API."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routes.campaigns import router as campaigns_router
from app.routes.brands import router as brands_router
from app.routes.ugc_studio import router as ugc_router
from app.services.asset_storage import AssetStorage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Sloth API started")
    yield
    logger.info("Sloth API shutting down")


# ── App ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Sloth API",
    description="AI Ad Creation Platform",
    version="0.1.0",
    lifespan=lifespan,
)

# ── Middleware ──────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ─────────────────────────────────────────────────────────────

app.include_router(brands_router)
app.include_router(campaigns_router)
app.include_router(ugc_router)

asset_storage = AssetStorage.from_settings()


@app.get("/assets/{asset_path:path}", include_in_schema=False)
async def serve_asset(asset_path: str) -> Response:
    stored_url = f"assets/{asset_path}".lstrip("/")
    try:
        data, content_type = await asset_storage.read_asset(stored_url)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Asset not found: {asset_path}") from exc

    return Response(content=data, media_type=content_type)


# ── Health check ────────────────────────────────────────────────────────

@app.get("/health", tags=["infra"])
async def health_check() -> dict[str, str]:
    return {"status": "ok"}
