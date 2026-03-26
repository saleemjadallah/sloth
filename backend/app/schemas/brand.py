"""Pydantic schemas for brand-related endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


# ── Request schemas ─────────────────────────────────────────────────────

class BrandAnalyzeRequest(BaseModel):
    """Body for POST /brands/analyze."""

    website_url: HttpUrl


class BrandAssetVariationRequest(BaseModel):
    """Request body for generating a prompted variation from an existing asset."""

    prompt: str = Field(min_length=5, max_length=2_000)


class BrandAssetGenerateRequest(BaseModel):
    """Request body for generating brand-fit assets from brand context."""

    prompt: str = Field(default="", max_length=2_000)
    count: int = Field(default=1, ge=1, le=4)


# ── Nested value objects ────────────────────────────────────────────────

class BrandColors(BaseModel):
    primary: str | None = None
    secondary: str | None = None
    accent: str | None = None

    @field_validator("primary", "secondary", "accent", mode="before")
    @classmethod
    def _coerce_color_value(cls, value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        if isinstance(value, (list, tuple, set)):
            for item in value:
                if isinstance(item, str) and item.strip():
                    return item.strip()
            return None
        return str(value)


class BrandFonts(BaseModel):
    heading: str | None = None
    body: str | None = None

    @field_validator("heading", "body", mode="before")
    @classmethod
    def _coerce_font_value(cls, value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        if isinstance(value, (list, tuple, set)):
            for item in value:
                if isinstance(item, str) and item.strip():
                    return item.strip()
            return None
        return str(value)


class BrandVoice(BaseModel):
    tone: str = ""
    style: str = ""
    personality_traits: list[str] = Field(default_factory=list)

    @field_validator("tone", "style", mode="before")
    @classmethod
    def _coerce_voice_text(cls, value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        return str(value)

    @field_validator("personality_traits", mode="before")
    @classmethod
    def _coerce_personality_traits(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            stripped = value.strip()
            return [stripped] if stripped else []
        return [str(value).strip()] if str(value).strip() else []


class BrandTargetAudience(BaseModel):
    demographics: str = ""
    pain_points: list[str] = Field(default_factory=list)
    desires: list[str] = Field(default_factory=list)

    @field_validator("demographics", mode="before")
    @classmethod
    def _coerce_demographics(cls, value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        return str(value)

    @field_validator("pain_points", "desires", mode="before")
    @classmethod
    def _coerce_audience_lists(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            stripped = value.strip()
            return [stripped] if stripped else []
        return [str(value).strip()] if str(value).strip() else []


class BrandProduct(BaseModel):
    name: str
    description: str = ""
    key_benefits: list[str] = Field(default_factory=list)


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
    saved_execution_count: int = 0
    published_execution_count: int = 0
    active_execution_status: str | None = None
    active_execution_updated_at: datetime | None = None
    active_execution_last_error: str | None = None
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
    saved_execution_count: int = 0
    published_execution_count: int = 0
    active_execution_status: str | None = None
    active_execution_updated_at: datetime | None = None
    active_execution_last_error: str | None = None
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
