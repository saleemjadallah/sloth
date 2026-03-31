"""Brand CRUD + analysis endpoints."""

from __future__ import annotations

import logging
import mimetypes
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from urllib.parse import urlsplit, urlunsplit

import httpx
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status
from fastapi.encoders import jsonable_encoder
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
    BrandAssetGenerateRequest,
    BrandAssetResponse,
    BrandAssetVariationRequest,
    BrandListItem,
    BrandProfile,
    BrandUpdate,
)
from app.schemas.creative import (
    BrandWorkspaceResponse,
    BrandWorkspaceUpdate,
    CreativeExecutionRequest,
    CreativeExecutionResponse,
    CreativeVideoRenderRequest,
    CreativeVideoRenderResponse,
    CreativeStudioResponse,
    LateAccountResponse,
    PublishSavedCreativeExecutionRequest,
    PublishSavedCreativeExecutionResponse,
    SavedCreativeExecutionCreate,
    SavedCreativeExecutionResponse,
    SavedCreativeExecutionSummary,
    VideoRenderRuntimeConfig,
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
from app.services.video_pipeline import (
    GoogleTTSService,
    MediaComposerService,
    MubertMusicService,
    VeoVideoService,
    VideoPipelineError,
    VideoPipelineService,
)

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


def _coerce_asset_text(value: object) -> str | None:
    """Normalize asset fields that must be stored as plain text."""
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, (list, tuple, set)):
        parts = [str(item).strip() for item in value if str(item).strip()]
        if not parts:
            return None
        return " | ".join(parts)
    if isinstance(value, dict):
        for key in ("text", "content", "description", "value", "url"):
            if key in value:
                return _coerce_asset_text(value.get(key))
        rendered = str(value).strip()
        return rendered or None
    rendered = str(value).strip()
    return rendered or None


def _coerce_asset_tags(value: object) -> list[str] | None:
    """Normalize asset tags to a list of strings."""
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        tags = [str(item).strip() for item in value if str(item).strip()]
        return tags or None
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else None
    rendered = str(value).strip()
    return [rendered] if rendered else None


def _clean_prompt_parts(values: object, *, limit: int | None = None) -> list[str]:
    """Normalize list-like prompt inputs to trimmed strings."""
    if isinstance(values, list):
        items = [str(item).strip() for item in values if str(item).strip()]
    elif values is None:
        items = []
    else:
        text = str(values).strip()
        items = [text] if text else []

    return items[:limit] if limit else items


def _summarize_brand_generation_prompt(brand: Brand, user_prompt: str) -> str:
    """Build a brand-context prompt for first-class asset generation."""
    voice = brand.voice or {}
    audience = brand.target_audience or {}
    colors = brand.colors or {}
    products = brand.products or []

    product_lines: list[str] = []
    for product in products[:3]:
        if not isinstance(product, dict):
            continue
        name = str(product.get("name") or "").strip()
        description = str(product.get("description") or "").strip()
        benefits = _clean_prompt_parts(product.get("key_benefits"), limit=3)
        if not name and not description and not benefits:
            continue
        detail_parts = [part for part in [name, description] if part]
        if benefits:
            detail_parts.append(f"benefits: {', '.join(benefits)}")
        product_lines.append("; ".join(detail_parts))

    palette = [
        f"{label} {value}".strip()
        for label, value in (
            ("primary", colors.get("primary")),
            ("secondary", colors.get("secondary")),
            ("accent", colors.get("accent")),
        )
        if value
    ]
    value_props = _clean_prompt_parts(brand.value_propositions, limit=4)
    pain_points = _clean_prompt_parts(audience.get("pain_points"), limit=3)
    desires = _clean_prompt_parts(audience.get("desires"), limit=3)

    prompt_parts = [
        f"Brand: {brand.name}.",
        f"Website: {brand.website_url}.",
        f"Industry: {brand.industry}." if brand.industry else "",
        (
            f"Voice and style: tone {voice.get('tone') or 'clear'}, style {voice.get('style') or 'confident'}."
            if voice
            else ""
        ),
        f"Brand palette: {', '.join(palette)}." if palette else "",
        f"Value propositions: {', '.join(value_props)}." if value_props else "",
        f"Target audience: {audience.get('demographics')}." if audience.get("demographics") else "",
        f"Audience pain points: {', '.join(pain_points)}." if pain_points else "",
        f"Audience desires: {', '.join(desires)}." if desires else "",
        f"Products or offers: {' | '.join(product_lines)}." if product_lines else "",
        (
            "Create a high-quality ad image that feels native to the brand and useful inside paid social creative workflows."
        ),
        (
            "If the business is not product-led, generate a strong service, lifestyle, editorial, or UI-led hero visual that still feels brand-specific."
        ),
        user_prompt.strip() and f"Additional creative direction: {user_prompt.strip()}.",
    ]
    return " ".join(part for part in prompt_parts if part)


def _guess_upload_content_type(upload: UploadFile) -> str:
    """Best-effort content-type detection for uploaded assets."""
    if upload.content_type and upload.content_type != "application/octet-stream":
        return upload.content_type
    return mimetypes.guess_type(upload.filename or "")[0] or "application/octet-stream"


def _is_supported_uploaded_asset(content_type: str) -> bool:
    """Return True for media types the studio can use directly."""
    return content_type.startswith("image/") or content_type.startswith("video/")


def _describe_uploaded_asset(upload: UploadFile) -> str:
    """Turn an uploaded file name into a UI-friendly description."""
    raw_name = Path(upload.filename or "uploaded asset").stem.replace("_", " ").replace("-", " ")
    return " ".join(raw_name.split()) or "Uploaded brand asset"


async def _persist_brand_asset(
    *,
    db: AsyncSession,
    brand_id: uuid.UUID,
    storage: AssetStorage,
    source_url: str,
    source_page: str | None = None,
    file_name: str,
    file_bytes: bytes,
    mime_type: str,
    width: int | None = None,
    height: int | None = None,
    category: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
    quality_score: int | None = None,
    is_usable: bool = True,
    alt_text: str | None = None,
    context: str | None = None,
    extraction_metadata: dict[str, object] | None = None,
) -> BrandAsset:
    """Persist a generated or uploaded asset into shared storage and the DB."""
    stored_url = await storage.save_asset(
        key=storage.build_key(str(brand_id), file_name),
        data=file_bytes,
        content_type=mime_type,
    )
    asset = BrandAsset(
        brand_id=brand_id,
        source_url=source_url,
        source_page=source_page,
        stored_url=stored_url,
        file_name=file_name,
        file_size=len(file_bytes),
        mime_type=mime_type,
        width=width,
        height=height,
        category=category,
        description=description,
        tags=tags,
        quality_score=quality_score,
        is_usable=is_usable,
        alt_text=alt_text,
        context=context,
        extraction_metadata=extraction_metadata,
    )
    db.add(asset)
    return asset


def _is_public_http_url(value: str | None) -> bool:
    """Return True when the value is a direct HTTP(S) URL."""
    if not value:
        return False
    lowered = value.lower()
    return lowered.startswith("http://") or lowered.startswith("https://")


def _infer_late_media_type(asset: BrandAsset) -> str | None:
    """Infer the media type expected by Late from the asset metadata."""
    mime_type = (asset.mime_type or "").lower()
    if mime_type.startswith("image/"):
        return "image"
    if mime_type.startswith("video/"):
        return "video"

    file_name = (asset.file_name or asset.stored_url or asset.source_url or "").lower()
    if file_name.endswith((".mp4", ".mov", ".m4v", ".webm")):
        return "video"
    if file_name.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp")):
        return "image"
    return None


def _build_public_asset_url(asset: BrandAsset, request: Request) -> str | None:
    """Return a publishable public URL for a brand asset."""
    if _is_public_http_url(asset.source_url):
        return asset.source_url
    if asset.stored_url:
        asset_path = asset.stored_url.lstrip("/").removeprefix("assets/")
        return str(request.url_for("serve_asset", asset_path=asset_path))
    return None


async def _build_late_media_items(
    record: CreativeExecution,
    db: AsyncSession,
    request: Request,
) -> list[dict[str, str]]:
    """Resolve selected execution assets into Late mediaItems payload entries."""
    concept = record.concept or {}
    raw_asset_ids = concept.get("asset_ids") if isinstance(concept, dict) else None
    asset_ids: list[uuid.UUID] = []
    for asset_id in raw_asset_ids or []:
        try:
            asset_ids.append(uuid.UUID(str(asset_id)))
        except (TypeError, ValueError):
            continue

    if not asset_ids:
        return []

    related_brand_ids = await _get_related_brand_ids(record.brand, db)
    assets_result = await db.execute(
        select(BrandAsset).where(
            BrandAsset.brand_id.in_(related_brand_ids),
            BrandAsset.id.in_(asset_ids),
        )
    )
    asset_map = {asset.id: asset for asset in assets_result.scalars().all()}

    media_items: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for asset_id in asset_ids:
        asset = asset_map.get(asset_id)
        if asset is None:
            continue
        media_type = _infer_late_media_type(asset)
        media_url = _build_public_asset_url(asset, request)
        if not media_type or not media_url or media_url in seen_urls:
            continue
        seen_urls.add(media_url)
        media_items.append({"type": media_type, "url": media_url})

    return media_items


def _default_workspace_delivery() -> dict[str, object]:
    """Return the default persisted delivery configuration."""
    return {
        "delivery_mode": "late_dev",
        "destination_label": None,
        "selected_late_profile_id": None,
        "publish_title": None,
        "content_override": None,
        "selected_late_account_ids": [],
        "publish_mode": "publish_now",
        "scheduled_for": None,
    }


def _sanitize_workspace_delivery(delivery: dict | None) -> dict[str, object]:
    """Normalize persisted delivery JSON to the API shape."""
    payload = {**_default_workspace_delivery(), **(delivery or {})}
    payload["selected_late_account_ids"] = [
        str(item).strip()
        for item in payload.get("selected_late_account_ids", [])
        if str(item).strip()
    ]
    return payload


def _sync_workspace_selection_from_studio(brand: Brand) -> None:
    """Keep workspace concept and asset selection aligned with the current studio snapshot."""
    studio = brand.workspace_studio or {}
    concepts = studio.get("concepts") if isinstance(studio, dict) else None
    if not isinstance(concepts, list) or not concepts:
        brand.workspace_selected_concept_id = None
        brand.workspace_selected_asset_ids = []
        return

    concept_ids = {
        str(concept.get("id"))
        for concept in concepts
        if isinstance(concept, dict) and concept.get("id")
    }
    selected_concept_id = brand.workspace_selected_concept_id
    if selected_concept_id not in concept_ids:
        first_concept = next(
            (concept for concept in concepts if isinstance(concept, dict) and concept.get("id")),
            None,
        )
        brand.workspace_selected_concept_id = (
            str(first_concept.get("id")) if first_concept else None
        )

    selected_concept = next(
        (
            concept
            for concept in concepts
            if isinstance(concept, dict)
            and str(concept.get("id")) == brand.workspace_selected_concept_id
        ),
        None,
    )
    selected_asset_ids = (
        selected_concept.get("asset_ids")
        if isinstance(selected_concept, dict)
        else None
    )
    brand.workspace_selected_asset_ids = [
        str(asset_id).strip()
        for asset_id in selected_asset_ids or []
        if str(asset_id).strip()
    ]


def _workspace_studio_has_reel(studio: dict | None) -> bool:
    """Return True when the persisted studio contains at least one reel concept."""
    if not isinstance(studio, dict):
        return False
    concepts = studio.get("concepts")
    if not isinstance(concepts, list):
        return False

    for concept in concepts:
        if not isinstance(concept, dict):
            continue
        signal = " ".join(
            str(concept.get(field) or "")
            for field in ("format", "name", "angle")
        ).lower()
        if any(keyword in signal for keyword in ("reel", "short-video", "short video", "video")):
            return True
    return False


def _serialize_workspace_execution(brand: Brand) -> dict[str, object] | None:
    """Return the currently persisted execution pack, if any."""
    execution = brand.workspace_execution
    if not isinstance(execution, dict):
        return None
    return execution


async def _build_brand_workspace_response(
    brand: Brand,
    db: AsyncSession,
) -> dict[str, object]:
    """Build the full persisted workspace payload for a brand/project."""
    related_brand_ids = await _get_related_brand_ids(brand, db)
    result = await db.execute(
        select(CreativeExecution)
        .where(CreativeExecution.brand_id.in_(related_brand_ids))
        .order_by(CreativeExecution.updated_at.desc(), CreativeExecution.created_at.desc())
    )
    records = result.scalars().all()
    service = _get_creative_studio_service()

    workspace_saved_execution_id = brand.workspace_saved_execution_id
    workspace_execution = _serialize_workspace_execution(brand)
    if workspace_saved_execution_id is not None:
        active_record = next(
            (record for record in records if record.id == workspace_saved_execution_id),
            None,
        )
        if active_record is not None:
            workspace_execution = service.serialize_saved_execution(active_record)["execution"]

    return {
        "brand": brand,
        "studio": brand.workspace_studio,
        "selected_concept_id": brand.workspace_selected_concept_id,
        "selected_asset_ids": brand.workspace_selected_asset_ids or [],
        "execution": workspace_execution,
        "saved_execution_id": workspace_saved_execution_id,
        "delivery": _sanitize_workspace_delivery(brand.workspace_delivery),
        "saved_executions": [
            service.serialize_saved_execution_summary(record) for record in records
        ],
    }

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


def _get_video_pipeline_service(runtime: VideoRenderRuntimeConfig) -> VideoPipelineService:
    """Build the integrated Veo/TTS/music pipeline from request-scoped runtime config."""
    project_id = runtime.project_id.strip()
    access_token = runtime.access_token.strip()
    gcs_bucket = runtime.gcs_bucket.strip()
    location = runtime.location.strip() or "us-central1"

    if not project_id or not access_token or not gcs_bucket:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Project ID, access token, and GCS bucket are required to render video.",
        )

    return VideoPipelineService(
        veo=VeoVideoService(
            project_id=project_id,
            access_token=access_token,
            gcs_bucket=gcs_bucket,
            location=location,
            default_model_id=settings.VEO_MODEL_ID,
        ),
        tts=GoogleTTSService(
            credentials_json=settings.GOOGLE_CREDENTIALS_JSON,
            default_voice_name=settings.TTS_VOICE_NAME,
            default_pitch=settings.TTS_PITCH,
            default_effects_profile_id=settings.TTS_EFFECTS_PROFILE_ID,
            max_script_chars=settings.TTS_MAX_SCRIPT_CHARS,
        ),
        music=MubertMusicService(
            company_id=settings.MUBERT_COMPANY_ID,
            license_token=settings.MUBERT_LICENSE_TOKEN,
        ),
        composer=MediaComposerService(),
        storage=AssetStorage.from_settings(),
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
            .options(
                selectinload(Brand.assets),
                selectinload(Brand.creative_executions),
            )
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
        .options(
            selectinload(Brand.assets),
            selectinload(Brand.creative_executions),
        )
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
                source_url=_coerce_asset_text(asset_data.get("source_url")) or "",
                source_page=_coerce_asset_text(asset_data.get("source_page")),
                stored_url=_coerce_asset_text(asset_data.get("stored_url")),
                file_name=_coerce_asset_text(asset_data.get("file_name")),
                file_size=asset_data.get("file_size"),
                mime_type=_coerce_asset_text(asset_data.get("mime_type")),
                width=asset_data.get("width"),
                height=asset_data.get("height"),
                category=_coerce_asset_text(asset_data.get("category")),
                description=_coerce_asset_text(asset_data.get("description")),
                tags=_coerce_asset_tags(asset_data.get("tags")),
                quality_score=asset_data.get("quality_score"),
                is_usable=asset_data.get("is_usable", True),
                alt_text=_coerce_asset_text(asset_data.get("alt_text")),
                context=_coerce_asset_text(asset_data.get("context")),
                extraction_metadata={
                    **(asset_data.get("extraction_metadata") or {}),
                    "suggested_ad_use": _coerce_asset_text(asset_data.get("suggested_ad_use")),
                },
            )
            db.add(asset)

    except Exception:
        logger.exception("Brand analysis failed for %s", website_url)
        brand.analysis_status = "failed"

    await db.flush()
    return await _get_brand_or_404(brand.id, db, with_assets=True)


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
        .options(
            selectinload(Brand.assets),
            selectinload(Brand.creative_executions),
        )
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
        .options(
            selectinload(Brand.assets),
            selectinload(Brand.creative_executions),
        )
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
    "/{brand_id}/workspace",
    response_model=BrandWorkspaceResponse,
    summary="Get the persisted project workspace for a brand",
)
async def get_brand_workspace(
    brand_id: uuid.UUID,
    concept_count: int = Query(4, ge=2, le=6, description="Number of concepts"),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    """Return the persisted creative workspace, generating the studio snapshot if needed."""
    brand = await _get_brand_or_404(brand_id, db, with_assets=True)
    if (
        not isinstance(brand.workspace_studio, dict)
        or brand.workspace_concept_count != concept_count
        or not _workspace_studio_has_reel(brand.workspace_studio)
    ):
        service = _get_creative_studio_service()
        studio = await service.build_studio(brand=brand, concept_count=concept_count)
        brand.workspace_studio = jsonable_encoder(studio)
        brand.workspace_concept_count = concept_count
        _sync_workspace_selection_from_studio(brand)
        await db.flush()
        brand = await _get_brand_or_404(brand.id, db, with_assets=True)

    return await _build_brand_workspace_response(brand, db)


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
    """Build or reuse a persisted creative brief and concept set for a brand."""
    brand = await _get_brand_or_404(brand_id, db, with_assets=True)
    if (
        isinstance(brand.workspace_studio, dict)
        and brand.workspace_concept_count == concept_count
        and _workspace_studio_has_reel(brand.workspace_studio)
    ):
        return brand.workspace_studio

    service = _get_creative_studio_service()
    studio = await service.build_studio(brand=brand, concept_count=concept_count)
    brand.workspace_studio = jsonable_encoder(studio)
    brand.workspace_concept_count = concept_count
    _sync_workspace_selection_from_studio(brand)
    await db.flush()
    return brand.workspace_studio


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
    execution = await service.build_execution_pack(
        brand=brand,
        brief=body.brief,
        concept=body.concept,
    )
    brand.workspace_selected_concept_id = body.concept.id
    brand.workspace_selected_asset_ids = list(body.concept.asset_ids)
    brand.workspace_execution = jsonable_encoder(execution)
    brand.workspace_saved_execution_id = None
    await db.flush()
    return execution


@router.post(
    "/{brand_id}/creative-execution/video-render",
    response_model=CreativeVideoRenderResponse,
    summary="Render a Veo/TTS/music video pipeline for the current execution pack",
)
async def render_brand_creative_execution_video(
    brand_id: uuid.UUID,
    body: CreativeVideoRenderRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    """Render the current execution pack into scene videos and a final composed asset."""
    brand = await _get_brand_or_404(brand_id, db, with_assets=True)
    selected_ids = {str(asset_id) for asset_id in body.selected_asset_ids}
    selected_assets = [
        {
            "id": str(asset.id),
            "stored_url": asset.stored_url,
            "source_url": asset.source_url,
            "mime_type": asset.mime_type,
            "description": asset.description,
            "category": asset.category,
        }
        for asset in (brand.assets or [])
        if str(asset.id) in selected_ids
    ]

    pipeline = _get_video_pipeline_service(body.runtime)
    render_session_id = f"video-render-{uuid.uuid4().hex[:12]}"
    execution_payload = body.execution.model_dump(mode="json")

    try:
        result = await pipeline.render_execution(
            brand_name=brand.name or brand.website_url,
            execution=execution_payload,
            selected_assets=selected_assets,
            storage_prefix=f"{brand_id}/{render_session_id}",
            regenerate_scenes=body.regenerate_scenes,
        )
    except VideoPipelineError as exc:
        logger.exception("Execution video render failed for brand %s", brand_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    created_assets: list[BrandAsset] = []
    render_state = {**result.state, "scene_artifacts": []}

    for index, artifact in enumerate(result.scene_videos, start=1):
        stored = await pipeline.persist_artifact(
            brand_id=str(brand.id),
            storage_prefix=f"{render_session_id}/scenes/{index}",
            artifact=artifact,
        )
        render_state["scene_artifacts"].append(
            {
                "kind": stored.kind,
                "label": stored.label,
                "stored_url": stored.stored_url,
                "asset_id": None,
                "mime_type": stored.mime_type,
                "file_name": stored.file_name,
                "source_gcs_uri": stored.source_gcs_uri,
                "scene_id": stored.scene_id,
            }
        )

    if result.stitched_video is not None:
        stored = await pipeline.persist_artifact(
            brand_id=str(brand.id),
            storage_prefix=f"{render_session_id}/stitched",
            artifact=result.stitched_video,
        )
        render_state["stitched_artifact"] = {
            "kind": stored.kind,
            "label": stored.label,
            "stored_url": stored.stored_url,
            "asset_id": None,
            "mime_type": stored.mime_type,
            "file_name": stored.file_name,
            "source_gcs_uri": stored.source_gcs_uri,
            "scene_id": stored.scene_id,
        }

    if result.voiceover is not None:
        stored = await pipeline.persist_artifact(
            brand_id=str(brand.id),
            storage_prefix=f"{render_session_id}/audio",
            artifact=result.voiceover,
        )
        render_state["voiceover_artifact"] = {
            "kind": stored.kind,
            "label": stored.label,
            "stored_url": stored.stored_url,
            "asset_id": None,
            "mime_type": stored.mime_type,
            "file_name": stored.file_name,
            "source_gcs_uri": stored.source_gcs_uri,
            "scene_id": stored.scene_id,
        }

    if result.music is not None:
        stored = await pipeline.persist_artifact(
            brand_id=str(brand.id),
            storage_prefix=f"{render_session_id}/audio",
            artifact=result.music,
        )
        render_state["music_artifact"] = {
            "kind": stored.kind,
            "label": stored.label,
            "stored_url": stored.stored_url,
            "asset_id": None,
            "mime_type": stored.mime_type,
            "file_name": stored.file_name,
            "source_gcs_uri": stored.source_gcs_uri,
            "scene_id": stored.scene_id,
        }

    if result.final_video is not None:
        final_asset = await _persist_brand_asset(
            db=db,
            brand_id=brand.id,
            storage=AssetStorage.from_settings(),
            source_url="generated://video-pipeline",
            file_name=f"{render_session_id}-final.mp4",
            file_bytes=result.final_video.data,
            mime_type=result.final_video.mime_type,
            category="generated-video",
            description=f"Rendered video for {body.execution.concept_name}",
            tags=["generated-video", "veo", "rendered-execution"],
            quality_score=9,
            is_usable=True,
            context=body.execution.video_brief.veo_prompt,
            extraction_metadata={
                "render_session_id": render_session_id,
                "render_settings": render_state.get("settings"),
                "render_provider": "vertex_veo",
            },
        )
        created_assets.append(final_asset)
        render_state["final_artifact"] = {
            "kind": result.final_video.kind,
            "label": result.final_video.label,
            "stored_url": final_asset.stored_url,
            "asset_id": str(final_asset.id),
            "mime_type": final_asset.mime_type or result.final_video.mime_type,
            "file_name": final_asset.file_name,
            "source_gcs_uri": result.final_video.source_gcs_uri,
            "scene_id": result.final_video.scene_id,
        }

    render_state["last_rendered_at"] = datetime.now(timezone.utc)
    execution_payload["video_render"] = render_state
    brand.workspace_execution = jsonable_encoder(execution_payload)
    if created_assets:
        existing_ids = [str(asset_id) for asset_id in (brand.workspace_selected_asset_ids or [])]
        existing_ids.extend(str(asset.id) for asset in created_assets)
        brand.workspace_selected_asset_ids = list(dict.fromkeys(existing_ids))
    await db.flush()
    for asset in created_assets:
        await db.refresh(asset)

    return {
        "execution": execution_payload,
        "created_assets": created_assets,
    }


@router.put(
    "/{brand_id}/workspace",
    response_model=BrandWorkspaceResponse,
    summary="Update the persisted project workspace for a brand",
)
async def update_brand_workspace(
    brand_id: uuid.UUID,
    body: BrandWorkspaceUpdate,
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    """Persist the current creative studio and delivery state for a brand workspace."""
    brand = await _get_brand_or_404(brand_id, db, with_assets=True)

    if body.studio is not None:
        brand.workspace_studio = body.studio.model_dump(mode="json")
        brand.workspace_concept_count = len(body.studio.concepts) or brand.workspace_concept_count

    if body.selected_concept_id is not None:
        brand.workspace_selected_concept_id = body.selected_concept_id
    brand.workspace_selected_asset_ids = list(body.selected_asset_ids)

    if body.execution is not None:
        brand.workspace_execution = body.execution.model_dump(mode="json")

    brand.workspace_saved_execution_id = body.saved_execution_id
    brand.workspace_delivery = body.delivery.model_dump(mode="json")
    await db.flush()
    brand = await _get_brand_or_404(brand.id, db, with_assets=True)
    return await _build_brand_workspace_response(brand, db)


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
        brief=body.brief.model_dump(mode="json"),
        concept=body.concept.model_dump(mode="json"),
        execution=body.execution.model_dump(mode="json"),
    )
    db.add(record)
    await db.flush()
    await db.refresh(record)

    brand.workspace_selected_concept_id = body.concept.id
    brand.workspace_selected_asset_ids = list(body.concept.asset_ids)
    brand.workspace_execution = body.execution.model_dump(mode="json")
    brand.workspace_saved_execution_id = record.id
    brand.workspace_delivery = {
        **_sanitize_workspace_delivery(brand.workspace_delivery),
        "delivery_mode": body.delivery_mode,
        "destination_label": body.destination_label,
    }
    await db.flush()

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
    request: Request,
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

    media_items = await _build_late_media_items(record, db, request)
    requires_media = any(
        (account.get("platform") or "").strip().lower() == "instagram"
        for account in selected_accounts
    )
    if requires_media and not media_items:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Instagram posts require at least one selected image or video asset.",
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
    if media_items:
        payload["mediaItems"] = media_items

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

    workspace_brand = record.brand or await _get_brand_or_404(brand_id, db, with_assets=True)
    workspace_brand.workspace_saved_execution_id = record.id
    workspace_brand.workspace_execution = jsonable_encoder(record.execution)
    workspace_brand.workspace_delivery = {
        **_sanitize_workspace_delivery(workspace_brand.workspace_delivery),
        "delivery_mode": record.delivery_mode,
        "destination_label": record.destination_label,
        "publish_title": body.title,
        "content_override": body.content_override,
        "selected_late_account_ids": list(body.account_ids),
        "publish_mode": body.mode,
        "scheduled_for": body.scheduled_for,
    }
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
    "/{brand_id}/assets/generate",
    response_model=list[BrandAssetResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Generate brand-fit visuals from the brand context",
)
async def generate_brand_assets(
    brand_id: uuid.UUID,
    body: BrandAssetGenerateRequest,
    db: AsyncSession = Depends(get_db),
) -> Sequence[BrandAsset]:
    """Create new AI-generated assets directly from brand strategy and product data."""
    brand = await _get_brand_or_404(brand_id, db)
    service = _get_image_variation_service()
    storage = AssetStorage.from_settings()
    prompt = _summarize_brand_generation_prompt(brand, body.prompt)

    try:
        outputs = await service.create_brand_assets(prompt=prompt, count=body.count)
    except Exception as exc:
        logger.exception("Brand asset generation failed for brand %s", brand_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Brand image generation failed: {exc}",
        ) from exc

    created_assets: list[BrandAsset] = []
    for index, output in enumerate(outputs, start=1):
        extension = {
            "image/png": ".png",
            "image/webp": ".webp",
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
        }.get(output["mime_type"].lower(), ".png")
        file_name = f"generated-brand-{uuid.uuid4().hex[:12]}-{index}{extension}"
        asset = await _persist_brand_asset(
            db=db,
            brand_id=brand_id,
            storage=storage,
            source_url="generated://brand-prompt",
            file_name=file_name,
            file_bytes=output["bytes"],
            mime_type=output["mime_type"],
            width=output["width"],
            height=output["height"],
            category="generated",
            description=f"AI-generated brand visual for {brand.name}",
            tags=["ai-generated", "brand-generated", "main-feature"],
            quality_score=9,
            is_usable=True,
            context=body.prompt.strip() or None,
            extraction_metadata={
                "generation_model": ImageVariationService.DEFAULT_MODEL,
                "generation_prompt": prompt,
                "requested_count": body.count,
            },
        )
        created_assets.append(asset)

    await db.flush()
    for asset in created_assets:
        await db.refresh(asset)
    return created_assets


@router.post(
    "/{brand_id}/assets/upload",
    response_model=list[BrandAssetResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Upload brand assets directly into the project",
)
async def upload_brand_assets(
    brand_id: uuid.UUID,
    files: list[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
) -> Sequence[BrandAsset]:
    """Persist uploaded image/video assets so they can be used inside Creative Studio."""
    await _get_brand_or_404(brand_id, db)
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Attach at least one file to upload brand assets.",
        )

    storage = AssetStorage.from_settings()
    created_assets: list[BrandAsset] = []

    for upload in files:
        content_type = _guess_upload_content_type(upload)
        if not _is_supported_uploaded_asset(content_type):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{upload.filename or 'Uploaded file'} must be an image or video asset.",
            )

        data = await upload.read()
        if not data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{upload.filename or 'Uploaded file'} is empty.",
            )

        suffix = Path(upload.filename or "").suffix
        guessed_suffix = mimetypes.guess_extension(content_type) or ""
        file_name = f"uploaded-{uuid.uuid4().hex[:12]}{suffix or guessed_suffix}"
        asset = await _persist_brand_asset(
            db=db,
            brand_id=brand_id,
            storage=storage,
            source_url=f"uploaded://{upload.filename or file_name}",
            file_name=file_name,
            file_bytes=data,
            mime_type=content_type,
            category="uploaded",
            description=_describe_uploaded_asset(upload),
            tags=["uploaded", "user-provided"],
            quality_score=9 if content_type.startswith("image/") else 8,
            is_usable=True,
            extraction_metadata={
                "upload_name": upload.filename or file_name,
                "content_type": content_type,
                "upload_source": "creative-studio",
            },
        )
        created_assets.append(asset)

    await db.flush()
    for asset in created_assets:
        await db.refresh(asset)
    return created_assets


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
        asset = await _persist_brand_asset(
            db=db,
            brand_id=brand_id,
            storage=storage,
            source_url=f"generated://{source_asset.id}",
            source_page=source_asset.source_page,
            file_name=file_name,
            file_bytes=output["bytes"],
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
