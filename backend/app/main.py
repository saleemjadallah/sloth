"""FastAPI application entry-point for the Sloth API."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routes.brands import router as brands_router

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

# ── Static files (serve downloaded brand assets) ────────────────────────

assets_dir = Path("assets")
assets_dir.mkdir(exist_ok=True)
app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")


# ── Health check ────────────────────────────────────────────────────────

@app.get("/health", tags=["infra"])
async def health_check() -> dict[str, str]:
    return {"status": "ok"}
