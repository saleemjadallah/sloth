"""Brand CRUD + analysis endpoints."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Sequence
from urllib.parse import urlsplit, urlunsplit

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.database import get_db
from app.models.brand import Brand
from app.models.brand_asset import BrandAsset
from app.models.creative_execution import CreativeExecution
from app.schemas.brand import (
    BrandAnalyzeRequest,
    BrandAssetResponse,
    BrandAssetVariationRequest,
    BrandListItem,
    BrandProfile,
    BrandUpdate,
)
from app.schemas.creative import (
    CreativeExecutionRequest,
    CreativeExecutionResponse,
    CreativeStudioResponse,
    LateAccountResponse,
    PublishSavedCreativeExecutionRequest,
    PublishSavedCreativeExecutionResponse,
    SavedCreativeExecutionCreate,
    SavedCreativeExecutionResponse,
    SavedCreativeExecutionSummary,
)
from app.services.asset_classifier import AssetClassifier
from app.services.asset_extractor import AssetExtractor
from app.services.brand_analysis import BrandAnalysisService
from app.services.asset_storage import AssetStorage
from app.services.creative_studio import CreativeStudioService
from app.services.firecrawl_service import FirecrawlService
from app.services.image_variation import ImageVariationService
from app.services.late_service import LateService
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/brands", tags=["brands"])


# ── Helpers / factories ─────────────────────────────────────────────────

def _normalize_brand_website_url(raw_url: str) -> str:
    """Normalize a website URL so repeat analyses reuse the same brand record."""
    candidate = raw_url.strip()
    if "://" not in candidate:
        candidate = f"https://{candidate}"

    parsed = urlsplit(candidate)
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    return urlunsplit((scheme, netloc, path, "", ""))


def _brand_lookup_tokens(raw_query: str) -> set[str]:
    """Build a small set of normalized lookup keys for brand resolution."""
    query = raw_query.strip().lower()
    if not query:
        return set()

    tokens = {query}
    normalized_url = _normalize_brand_website_url(query)
    parsed = urlsplit(normalized_url)
    domain = parsed.netloc.lower().removeprefix("www.")

    tokens.add(normalized_url.lower())
    tokens.add(domain)
    if domain:
        tokens.add(f"https://{domain}")
        tokens.add(f"http://{domain}")
    return {token for token in tokens if token}


def _dedupe_brands(brands: Sequence[Brand]) -> list[Brand]:
    """Return the newest brand per normalized website URL."""
    deduped: list[Brand] = []
    seen_urls: set[str] = set()

    for brand in brands:
        normalized_url = _normalize_brand_website_url(brand.website_url)
        if normalized_url in seen_urls:
            continue
        seen_urls.add(normalized_url)
        deduped.append(brand)

    return deduped

def _get_analysis_service() -> BrandAnalysisService:
    """Build the analysis service with current config."""
    firecrawl = FirecrawlService(api_key=settings.FIRECRAWL_API_KEY)
    llm = LLMService(anthropic_api_key=settings.ANTHROPIC_API_KEY)
    storage = AssetStorage.from_settings()
    extractor = AssetExtractor(firecrawl_service=firecrawl, storage=storage)
    classifier = AssetClassifier(
        anthropic_api_key=settings.ANTHROPIC_API_KEY,
        storage=storage,
    )
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


def _get_image_variation_service() -> ImageVariationService:
    """Build the Gemini-powered asset variation service."""
    if not settings.GOOGLE_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Image variation is not configured. Set GOOGLE_API_KEY.",
        )
    return ImageVariationService(api_key=settings.GOOGLE_API_KEY)


def _get_late_service() -> LateService:
    """Build the Late publishing service using current config."""
    api_key = settings.LATE_API_KEY or settings.ZERNIO_API_KEY
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Late integration is not configured. Set LATE_API_KEY.",
        )
    return LateService(
        api_key=api_key,
        base_url=settings.LATE_API_BASE_URL,
    )


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


async def _get_saved_execution_or_404(
    brand_id: uuid.UUID,
    execution_id: uuid.UUID,
    db: AsyncSession,
) -> CreativeExecution:
    """Fetch a saved execution by ID and brand or raise 404."""
    brand = await _get_brand_or_404(brand_id, db)
    related_brand_ids = await _get_related_brand_ids(brand, db)
    result = await db.execute(
        select(CreativeExecution)
        .options(selectinload(CreativeExecution.brand))
        .where(
            CreativeExecution.id == execution_id,
            CreativeExecution.brand_id.in_(related_brand_ids),
        )
    )
    execution = result.scalar_one_or_none()
    if execution is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Creative execution {execution_id} not found for brand {brand_id}.",
        )
    return execution


async def _get_related_brand_ids(
    brand: Brand,
    db: AsyncSession,
) -> list[uuid.UUID]:
    """Return all brand IDs that represent the same normalized website."""
    result = await db.execute(select(Brand.id, Brand.website_url))
    normalized_url = _normalize_brand_website_url(brand.website_url)
    related_ids = [
        row.id
        for row in result.all()
        if _normalize_brand_website_url(row.website_url) == normalized_url
    ]
    return related_ids or [brand.id]


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
    website_url = _normalize_brand_website_url(str(body.website_url))

    existing_result = await db.execute(
        select(Brand)
        .options(selectinload(Brand.assets))
        .order_by(Brand.updated_at.desc(), Brand.created_at.desc())
    )
    brand = next(
        (
            candidate
            for candidate in existing_result.scalars().all()
            if _normalize_brand_website_url(candidate.website_url) == website_url
        ),
        None,
    )

    if brand is None:
        # Create a placeholder record so the client can poll status.
        brand = Brand(
            name="",
            website_url=website_url,
            analysis_status="analyzing",
        )
        db.add(brand)
        await db.flush()  # assigns the id
    else:
        brand.website_url = website_url
        brand.analysis_status = "analyzing"
        await db.flush()

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

        # Replace extracted assets so re-analysis refreshes the same brand record.
        await db.execute(delete(BrandAsset).where(BrandAsset.brand_id == brand.id))

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
        select(Brand)
        .options(selectinload(Brand.assets))
        .order_by(Brand.updated_at.desc(), Brand.created_at.desc())
    )
    return _dedupe_brands(result.scalars().all())


@router.get(
    "/lookup",
    response_model=BrandListItem,
    summary="Find the latest saved brand by domain, URL, or brand name",
)
async def lookup_brand(
    query: str = Query(..., min_length=2, description="Domain, URL, or brand name"),
    db: AsyncSession = Depends(get_db),
) -> Brand:
    """Resolve a user-entered brand query to the latest saved brand profile."""
    lookup_tokens = _brand_lookup_tokens(query)
    result = await db.execute(
        select(Brand)
        .options(selectinload(Brand.assets))
        .order_by(Brand.updated_at.desc(), Brand.created_at.desc())
    )
    brands = _dedupe_brands(result.scalars().all())

    def exact_match(brand: Brand) -> bool:
        normalized_url = _normalize_brand_website_url(brand.website_url).lower()
        domain = urlsplit(normalized_url).netloc.lower().removeprefix("www.")
        name = (brand.name or "").strip().lower()
        return bool({normalized_url, domain, name} & lookup_tokens)

    def partial_match(brand: Brand) -> bool:
        normalized_url = _normalize_brand_website_url(brand.website_url).lower()
        domain = urlsplit(normalized_url).netloc.lower().removeprefix("www.")
        name = (brand.name or "").strip().lower()
        return any(
            token in normalized_url or token in domain or token in name
            for token in lookup_tokens
        )

    match = next((brand for brand in brands if exact_match(brand)), None)
    if match is None:
        match = next((brand for brand in brands if partial_match(brand)), None)

    if match is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No saved brand found for '{query}'.",
        )
    return match


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


@router.post(
    "/{brand_id}/creative-execution",
    response_model=CreativeExecutionResponse,
    summary="Expand a concept into copy, design, and video outputs",
)
async def create_brand_creative_execution(
    brand_id: uuid.UUID,
    body: CreativeExecutionRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    """Build an execution pack for a selected concept."""
    brand = await _get_brand_or_404(brand_id, db, with_assets=True)
    service = _get_creative_studio_service()
    return await service.build_execution_pack(
        brand=brand,
        brief=body.brief,
        concept=body.concept,
    )


@router.post(
    "/{brand_id}/creative-executions",
    response_model=SavedCreativeExecutionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Save a generated execution pack",
)
async def save_brand_creative_execution(
    brand_id: uuid.UUID,
    body: SavedCreativeExecutionCreate,
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    """Persist an execution pack for export and publishing workflows."""
    brand = await _get_brand_or_404(brand_id, db, with_assets=True)

    record = CreativeExecution(
        brand_id=brand.id,
        concept_id=body.concept.id,
        concept_name=body.concept.name,
        summary=body.execution.summary,
        delivery_mode=body.delivery_mode,
        status=body.status,
        destination_label=body.destination_label,
        brief=body.brief.model_dump(),
        concept=body.concept.model_dump(),
        execution=body.execution.model_dump(),
    )
    db.add(record)
    await db.flush()
    await db.refresh(record)

    service = _get_creative_studio_service()
    return service.serialize_saved_execution(record)


@router.get(
    "/{brand_id}/creative-executions",
    response_model=list[SavedCreativeExecutionSummary],
    summary="List saved execution packs for a brand",
)
async def list_brand_creative_executions(
    brand_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, object]]:
    """Return saved execution packs ordered by most recent first."""
    brand = await _get_brand_or_404(brand_id, db)
    related_brand_ids = await _get_related_brand_ids(brand, db)
    result = await db.execute(
        select(CreativeExecution)
        .where(CreativeExecution.brand_id.in_(related_brand_ids))
        .order_by(CreativeExecution.updated_at.desc(), CreativeExecution.created_at.desc())
    )
    records = result.scalars().all()
    service = _get_creative_studio_service()
    return [service.serialize_saved_execution_summary(record) for record in records]


@router.get(
    "/{brand_id}/creative-executions/{execution_id}",
    response_model=SavedCreativeExecutionResponse,
    summary="Get a saved execution pack",
)
async def get_brand_creative_execution(
    brand_id: uuid.UUID,
    execution_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    """Return one persisted execution pack."""
    record = await _get_saved_execution_or_404(brand_id, execution_id, db)
    service = _get_creative_studio_service()
    return service.serialize_saved_execution(record)


@router.get(
    "/{brand_id}/creative-executions/{execution_id}/export",
    summary="Export a saved execution pack as markdown, text, or JSON",
    response_model=None,
)
async def export_brand_creative_execution(
    brand_id: uuid.UUID,
    execution_id: uuid.UUID,
    export_format: str = Query("markdown", alias="format", pattern="^(markdown|txt|json)$"),
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Render a saved execution as a downloadable document."""
    record = await _get_saved_execution_or_404(brand_id, execution_id, db)
    service = _get_creative_studio_service()
    content, media_type, filename = service.build_export_document(record, export_format)
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get(
    "/{brand_id}/late/accounts",
    response_model=list[LateAccountResponse],
    summary="List connected Late social accounts",
)
async def list_brand_late_accounts(
    brand_id: uuid.UUID,
    profile_id: str | None = Query(None, alias="profileId"),
    platform: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, object]]:
    """Return connected Late accounts available to the API key."""
    await _get_brand_or_404(brand_id, db)
    late = _get_late_service()

    try:
        accounts = await late.list_accounts(profile_id=profile_id, platform=platform)
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text or "Late account lookup failed."
        raise HTTPException(status_code=exc.response.status_code, detail=detail) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Late account lookup failed: {exc}",
        ) from exc

    normalized_accounts = []
    for account in accounts:
        profile = account.get("profileId")
        normalized_accounts.append(
            {
                "id": account.get("_id") or account.get("id") or "",
                "platform": account.get("platform") or "",
                "username": account.get("username"),
                "display_name": account.get("displayName"),
                "profile_url": account.get("profileUrl"),
                "is_active": bool(account.get("isActive", True)),
                "profile": (
                    {
                        "id": profile.get("_id") or profile.get("id") or "",
                        "name": profile.get("name") or "",
                        "slug": profile.get("slug"),
                    }
                    if isinstance(profile, dict)
                    else None
                ),
            }
        )
    return normalized_accounts


@router.post(
    "/{brand_id}/creative-executions/{execution_id}/publish",
    response_model=PublishSavedCreativeExecutionResponse,
    summary="Send a saved execution to Late for draft, scheduling, or publish-now",
)
async def publish_brand_creative_execution(
    brand_id: uuid.UUID,
    execution_id: uuid.UUID,
    body: PublishSavedCreativeExecutionRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    """Publish a saved execution via Late."""
    if not body.account_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one Late account must be selected.",
        )
    if body.mode == "schedule" and body.scheduled_for is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="scheduled_for is required when mode is schedule.",
        )

    record = await _get_saved_execution_or_404(brand_id, execution_id, db)
    late = _get_late_service()

    try:
        accounts = await late.list_accounts()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text or "Late account lookup failed."
        raise HTTPException(status_code=exc.response.status_code, detail=detail) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Late account lookup failed: {exc}",
        ) from exc

    selected_accounts = [
        account for account in accounts if (account.get("_id") or account.get("id")) in set(body.account_ids)
    ]
    if len(selected_accounts) != len(set(body.account_ids)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="One or more selected Late accounts could not be found.",
        )

    execution = record.execution or {}
    content = (
        body.content_override
        or "\n\n".join(
            part
            for part in [
                (execution.get("headlines") or [record.concept_name])[0],
                (execution.get("primary_text_variants") or [record.summary])[0],
                f"CTA: {(execution.get('ctas') or ['Learn More'])[0]}",
            ]
            if part
        )
    )

    payload: dict[str, object] = {
        "title": body.title or record.concept_name,
        "content": content,
        "platforms": [
            {
                "platform": account.get("platform"),
                "accountId": account.get("_id") or account.get("id"),
            }
            for account in selected_accounts
        ],
        "timezone": body.timezone,
        "metadata": {
            "brandId": str(record.brand_id),
            "creativeExecutionId": str(record.id),
            "conceptId": record.concept_id,
            "deliveryMode": record.delivery_mode,
        },
    }

    if body.mode == "draft":
        payload["isDraft"] = True
    elif body.mode == "publish_now":
        payload["publishNow"] = True
    elif body.mode == "schedule" and body.scheduled_for is not None:
        payload["scheduledFor"] = body.scheduled_for.astimezone(timezone.utc).isoformat()

    try:
        remote = await late.create_post(payload)
    except httpx.HTTPStatusError as exc:
        record.status = "failed"
        record.last_publish_error = exc.response.text or "Late publish failed."
        await db.flush()
        detail = exc.response.text or "Late publish failed."
        raise HTTPException(status_code=exc.response.status_code, detail=detail) from exc
    except httpx.HTTPError as exc:
        record.status = "failed"
        record.last_publish_error = str(exc)
        await db.flush()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Late publish failed: {exc}",
        ) from exc

    post = remote.get("post", {}) if isinstance(remote, dict) else {}
    platforms = post.get("platforms", []) if isinstance(post, dict) else []
    first_platform = platforms[0] if platforms else {}
    remote_status = post.get("status") or ("draft" if body.mode == "draft" else None)

    record.external_post_id = post.get("_id")
    record.external_post_url = first_platform.get("platformPostUrl")
    record.last_publish_error = None
    record.publishing_metadata = remote
    record.scheduled_for = body.scheduled_for if body.mode == "schedule" else None
    record.published_at = datetime.now(timezone.utc) if body.mode == "publish_now" else None
    if body.mode == "draft":
        record.status = "draft"
    elif body.mode == "schedule":
        record.status = "queued"
    else:
        record.status = "published"
    await db.flush()
    await db.refresh(record)

    service = _get_creative_studio_service()
    saved_execution = service.serialize_saved_execution(record)
    return {
        "saved_execution": saved_execution,
        "remote_post_id": record.external_post_id,
        "remote_post_status": remote_status,
        "remote_post_url": record.external_post_url,
        "message": remote.get("message", "Post synced with Late.") if isinstance(remote, dict) else "Post synced with Late.",
    }


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


@router.post(
    "/{brand_id}/assets/{asset_id}/variations",
    response_model=list[BrandAssetResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Generate prompted visual variations from an existing asset",
)
async def create_brand_asset_variations(
    brand_id: uuid.UUID,
    asset_id: uuid.UUID,
    body: BrandAssetVariationRequest,
    db: AsyncSession = Depends(get_db),
) -> Sequence[BrandAsset]:
    """Create AI-generated brand assets derived from an existing source asset."""
    await _get_brand_or_404(brand_id, db)
    source_asset = await db.get(BrandAsset, asset_id)
    if source_asset is None or source_asset.brand_id != brand_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Asset {asset_id} not found for brand {brand_id}.",
        )
    if not source_asset.stored_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Selected asset is missing a stored image and cannot be used for variation generation.",
        )

    storage = AssetStorage.from_settings()

    try:
        source_bytes, _ = await storage.read_asset(
            source_asset.stored_url,
            fallback_content_type=source_asset.mime_type,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not read source asset from storage: {exc}",
        ) from exc

    service = _get_image_variation_service()

    try:
        outputs = await service.create_variation(
            source_bytes=source_bytes,
            source_mime_type=source_asset.mime_type or "image/png",
            prompt=body.prompt,
        )
    except Exception as exc:
        logger.exception(
            "Asset variation generation failed for brand %s asset %s",
            brand_id,
            asset_id,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Asset variation generation failed: {exc}",
        ) from exc

    created_assets: list[BrandAsset] = []
    for output in outputs:
        extension = {
            "image/png": ".png",
            "image/webp": ".webp",
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
        }.get(output["mime_type"].lower(), ".png")
        file_name = f"generated-{uuid.uuid4().hex[:12]}{extension}"
        stored_url = await storage.save_asset(
            key=storage.build_key(str(brand_id), file_name),
            data=output["bytes"],
            content_type=output["mime_type"],
        )
        asset = BrandAsset(
            brand_id=brand_id,
            source_url=f"generated://{source_asset.id}",
            source_page=source_asset.source_page,
            stored_url=stored_url,
            file_name=file_name,
            file_size=len(output["bytes"]),
            mime_type=output["mime_type"],
            width=output["width"],
            height=output["height"],
            category=source_asset.category or "generated",
            description=f"AI variation of {source_asset.description or source_asset.category or 'brand asset'}",
            tags=["ai-generated", "nano-banana-pro"],
            quality_score=max(source_asset.quality_score or 7, 8),
            is_usable=True,
            alt_text=source_asset.alt_text,
            context=body.prompt,
            extraction_metadata={
                "generation_model": ImageVariationService.DEFAULT_MODEL,
                "generation_prompt": body.prompt,
                "source_asset_id": str(source_asset.id),
                "source_stored_url": source_asset.stored_url,
                "source_source_url": source_asset.source_url,
            },
        )
        db.add(asset)
        created_assets.append(asset)

    await db.flush()
    for asset in created_assets:
        await db.refresh(asset)
    return created_assets


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
