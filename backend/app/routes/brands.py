"""Brand CRUD + analysis endpoints."""

from __future__ import annotations

import uuid
import logging
from typing import Sequence

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.database import get_db
from app.models.brand import Brand
from app.models.brand_asset import BrandAsset
from app.schemas.brand import (
    BrandAnalyzeRequest,
    BrandAssetResponse,
    BrandListItem,
    BrandProfile,
    BrandUpdate,
)
from app.schemas.creative import CreativeStudioResponse
from app.services.asset_classifier import AssetClassifier
from app.services.asset_extractor import AssetExtractor
from app.services.brand_analysis import BrandAnalysisService
from app.services.creative_studio import CreativeStudioService
from app.services.firecrawl_service import FirecrawlService
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/brands", tags=["brands"])


# ── Helpers / factories ─────────────────────────────────────────────────

def _get_analysis_service() -> BrandAnalysisService:
    """Build the analysis service with current config."""
    firecrawl = FirecrawlService(api_key=settings.FIRECRAWL_API_KEY)
    llm = LLMService(anthropic_api_key=settings.ANTHROPIC_API_KEY)
    extractor = AssetExtractor(firecrawl_service=firecrawl, storage_dir="assets")
    classifier = AssetClassifier(anthropic_api_key=settings.ANTHROPIC_API_KEY)
    return BrandAnalysisService(
        firecrawl_service=firecrawl,
        llm_service=llm,
        asset_extractor=extractor,
        asset_classifier=classifier,
    )


def _get_creative_studio_service() -> CreativeStudioService:
    """Build the creative studio service with optional LLM support."""
    llm = None
    if settings.ANTHROPIC_API_KEY:
        llm = LLMService(anthropic_api_key=settings.ANTHROPIC_API_KEY)
    return CreativeStudioService(llm_service=llm)


async def _get_brand_or_404(
    brand_id: uuid.UUID,
    db: AsyncSession,
    *,
    with_assets: bool = False,
) -> Brand:
    """Fetch a brand by ID or raise 404."""
    if with_assets:
        result = await db.execute(
            select(Brand)
            .options(selectinload(Brand.assets))
            .where(Brand.id == brand_id)
        )
        brand = result.scalar_one_or_none()
    else:
        brand = await db.get(Brand, brand_id)
    if brand is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Brand {brand_id} not found.",
        )
    return brand


# ── Endpoints ───────────────────────────────────────────────────────────

@router.post(
    "/analyze",
    response_model=BrandProfile,
    status_code=status.HTTP_201_CREATED,
    summary="Analyze a website and create a brand profile with extracted assets",
)
async def analyze_brand(
    body: BrandAnalyzeRequest,
    db: AsyncSession = Depends(get_db),
) -> Brand:
    """Scrape the given URL, run LLM analysis, extract assets, and persist."""
    website_url = str(body.website_url)

    # Create a placeholder record so the client can poll status.
    brand = Brand(
        name="",
        website_url=website_url,
        analysis_status="analyzing",
    )
    db.add(brand)
    await db.flush()  # assigns the id

    try:
        service = _get_analysis_service()
        profile = await service.analyze(
            website_url=website_url,
            brand_id=str(brand.id),
        )

        brand.name = profile.get("name") or website_url
        brand.logo_url = profile.get("logo_url")
        brand.colors = profile.get("colors")
        brand.fonts = profile.get("fonts")
        brand.voice = profile.get("voice")
        brand.value_propositions = profile.get("value_propositions")
        brand.target_audience = profile.get("target_audience")
        brand.products = profile.get("products")
        brand.industry = profile.get("industry")
        brand.raw_analysis = profile.get("raw_analysis")
        brand.analysis_status = "completed"

        # Persist extracted assets
        for asset_data in profile.get("assets", []):
            asset = BrandAsset(
                brand_id=brand.id,
                source_url=asset_data.get("source_url", ""),
                source_page=asset_data.get("source_page"),
                stored_url=asset_data.get("stored_url"),
                file_name=asset_data.get("file_name"),
                file_size=asset_data.get("file_size"),
                mime_type=asset_data.get("mime_type"),
                width=asset_data.get("width"),
                height=asset_data.get("height"),
                category=asset_data.get("category"),
                description=asset_data.get("description"),
                tags=asset_data.get("tags"),
                quality_score=asset_data.get("quality_score"),
                is_usable=asset_data.get("is_usable", True),
                alt_text=asset_data.get("alt_text"),
                context=asset_data.get("context"),
                extraction_metadata={
                    "suggested_ad_use": asset_data.get("suggested_ad_use"),
                },
            )
            db.add(asset)

    except Exception:
        logger.exception("Brand analysis failed for %s", website_url)
        brand.analysis_status = "failed"

    await db.flush()
    await db.refresh(brand)
    return brand


@router.get(
    "/",
    response_model=list[BrandListItem],
    summary="List all brands",
)
async def list_brands(
    db: AsyncSession = Depends(get_db),
) -> Sequence[Brand]:
    """Return all brands ordered by creation date (newest first)."""
    result = await db.execute(
        select(Brand).order_by(Brand.created_at.desc())
    )
    return result.scalars().all()


@router.get(
    "/{brand_id}",
    response_model=BrandProfile,
    summary="Get a single brand profile",
)
async def get_brand(
    brand_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> Brand:
    return await _get_brand_or_404(brand_id, db, with_assets=True)


@router.get(
    "/{brand_id}/creative-studio",
    response_model=CreativeStudioResponse,
    summary="Generate a creative brief and ad concepts for a brand",
)
async def get_brand_creative_studio(
    brand_id: uuid.UUID,
    concept_count: int = Query(4, ge=2, le=6, description="Number of concepts"),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    """Build a creative brief and concept set from the stored brand profile."""
    brand = await _get_brand_or_404(brand_id, db, with_assets=True)
    service = _get_creative_studio_service()
    return await service.build_studio(brand=brand, concept_count=concept_count)


@router.get(
    "/{brand_id}/assets",
    response_model=list[BrandAssetResponse],
    summary="Get all assets for a brand",
)
async def get_brand_assets(
    brand_id: uuid.UUID,
    category: str | None = Query(None, description="Filter by category"),
    usable_only: bool = Query(False, description="Only return usable assets"),
    min_quality: int = Query(0, ge=0, le=10, description="Minimum quality score"),
    db: AsyncSession = Depends(get_db),
) -> Sequence[BrandAsset]:
    """Return assets for a brand, optionally filtered."""
    await _get_brand_or_404(brand_id, db)  # ensure brand exists

    query = select(BrandAsset).where(BrandAsset.brand_id == brand_id)

    if category:
        query = query.where(BrandAsset.category == category)
    if usable_only:
        query = query.where(BrandAsset.is_usable.is_(True))
    if min_quality > 0:
        query = query.where(BrandAsset.quality_score >= min_quality)

    query = query.order_by(BrandAsset.quality_score.desc().nulls_last())
    result = await db.execute(query)
    return result.scalars().all()


@router.put(
    "/{brand_id}",
    response_model=BrandProfile,
    summary="Update a brand profile",
)
async def update_brand(
    brand_id: uuid.UUID,
    body: BrandUpdate,
    db: AsyncSession = Depends(get_db),
) -> Brand:
    brand = await _get_brand_or_404(brand_id, db)

    update_data = body.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        if hasattr(value, "model_dump"):
            value = value.model_dump()
        elif isinstance(value, list):
            value = [
                item.model_dump() if hasattr(item, "model_dump") else item
                for item in value
            ]
        setattr(brand, field, value)

    await db.flush()
    await db.refresh(brand)
    return brand


@router.delete(
    "/{brand_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a brand",
    response_model=None,
)
async def delete_brand(
    brand_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> None:
    brand = await _get_brand_or_404(brand_id, db)
    await db.delete(brand)
    await db.flush()
