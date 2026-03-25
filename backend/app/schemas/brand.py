"""Pydantic schemas for brand-related endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, HttpUrl


# ── Request schemas ─────────────────────────────────────────────────────

class BrandAnalyzeRequest(BaseModel):
    """Body for POST /brands/analyze."""

    website_url: HttpUrl


# ── Nested value objects ────────────────────────────────────────────────

class BrandColors(BaseModel):
    primary: str | None = None
    secondary: str | None = None
    accent: str | None = None


class BrandFonts(BaseModel):
    heading: str | None = None
    body: str | None = None


class BrandVoice(BaseModel):
    tone: str = ""
    style: str = ""
    personality_traits: list[str] = []


class BrandTargetAudience(BaseModel):
    demographics: str = ""
    pain_points: list[str] = []
    desires: list[str] = []


class BrandProduct(BaseModel):
    name: str
    description: str = ""
    key_benefits: list[str] = []


# ── Asset schemas ──────────────────────────────────────────────────────

class BrandAssetResponse(BaseModel):
    """Asset extracted from a brand's website."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    brand_id: uuid.UUID
    source_url: str
    source_page: str | None = None
    stored_url: str | None = None
    file_name: str | None = None
    file_size: int | None = None
    mime_type: str | None = None
    width: int | None = None
    height: int | None = None
    category: str | None = None
    description: str | None = None
    tags: list[str] | None = None
    quality_score: int | None = None
    is_usable: bool = True
    alt_text: str | None = None
    context: str | None = None
    extraction_metadata: dict | None = None
    created_at: datetime


class BrandAssetSummary(BaseModel):
    """Lightweight asset summary for inclusion in brand profile."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    source_url: str
    stored_url: str | None = None
    category: str | None = None
    description: str | None = None
    quality_score: int | None = None
    is_usable: bool = True
    width: int | None = None
    height: int | None = None


# ── Response schemas ────────────────────────────────────────────────────

class BrandProfile(BaseModel):
    """Full brand profile returned from the API."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: str | None = None
    name: str
    website_url: str
    logo_url: str | None = None
    colors: BrandColors | None = None
    fonts: BrandFonts | None = None
    voice: BrandVoice | None = None
    value_propositions: list[str] | None = None
    target_audience: BrandTargetAudience | None = None
    products: list[BrandProduct] | None = None
    industry: str | None = None
    raw_analysis: dict | None = None
    analysis_status: str = "pending"
    assets: list[BrandAssetSummary] = []
    asset_count: int = 0
    usable_asset_count: int = 0
    created_at: datetime
    updated_at: datetime


class BrandListItem(BaseModel):
    """Lightweight brand item for list endpoints."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    website_url: str
    logo_url: str | None = None
    industry: str | None = None
    analysis_status: str
    asset_count: int = 0
    created_at: datetime


# ── Update schema ───────────────────────────────────────────────────────

class BrandUpdate(BaseModel):
    """Partial update — every field is optional."""

    name: str | None = None
    website_url: str | None = None
    logo_url: str | None = None
    colors: BrandColors | None = None
    fonts: BrandFonts | None = None
    voice: BrandVoice | None = None
    value_propositions: list[str] | None = None
    target_audience: BrandTargetAudience | None = None
    products: list[BrandProduct] | None = None
    industry: str | None = None
