"""UGC Studio endpoints — completely isolated from the creative studio pipeline."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.brand import Brand
from app.schemas.ugc import (
    UgcAvatar,
    UgcAvatarLibraryResponse,
    UgcGenerateScriptRequest,
    UgcGenerateScriptResponse,
    UgcGenerateVideoRequest,
    UgcGenerateVideoResponse,
    UgcJobState,
    UgcJobStatusResponse,
    UgcPipelineStep,
)
from app.services.asset_storage import AssetStorage
from app.services.ugc_pipeline import FalAIService, UgcPipelineService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ugc", tags=["ugc-studio"])

# ── In-memory job store (MVP — replace with Redis for production) ───────

_job_store: dict[str, UgcJobState] = {}


# ── Service factories ───────────────────────────────────────────────────


def _build_ugc_pipeline() -> UgcPipelineService:
    from app.services.llm_service import LLMService
    from app.services.video_pipeline import (
        GoogleTTSService,
        MediaComposerService,
        MubertMusicService,
        VeoVideoService,
        VideoPipelineService,
    )

    fal = FalAIService(api_key=settings.FAL_API_KEY)
    tts = GoogleTTSService(
        credentials_json=settings.GOOGLE_CREDENTIALS_JSON,
        default_voice_name=settings.TTS_VOICE_NAME,
        default_pitch=settings.TTS_PITCH,
        default_effects_profile_id=settings.TTS_EFFECTS_PROFILE_ID,
        max_script_chars=settings.TTS_MAX_SCRIPT_CHARS,
    )
    music = MubertMusicService(
        company_id=settings.MUBERT_COMPANY_ID,
        license_token=settings.MUBERT_LICENSE_TOKEN,
    )
    composer = MediaComposerService()
    storage = AssetStorage.from_settings()
    llm = LLMService(settings.ANTHROPIC_API_KEY) if settings.ANTHROPIC_API_KEY else None
    veo_pipeline = VideoPipelineService(
        veo=VeoVideoService(
            project_id=settings.VEO_PROJECT_ID,
            access_token=settings.VEO_ACCESS_TOKEN,
            gcs_bucket=settings.VEO_GCS_BUCKET,
            location=settings.VEO_LOCATION,
            default_model_id=settings.VEO_MODEL_ID,
        ),
        tts=tts,
        music=music,
        composer=composer,
        storage=storage,
    )

    return UgcPipelineService(
        fal=fal,
        tts=tts,
        music=music,
        composer=composer,
        storage=storage,
        llm=llm,
        veo_pipeline=veo_pipeline,
    )


async def _get_brand_or_404(brand_id: uuid.UUID, db: AsyncSession) -> Brand:
    brand = await db.get(Brand, brand_id)
    if brand is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Brand {brand_id} not found.",
        )
    return brand


# ── Built-in avatar library ─────────────────────────────────────────────

AVATAR_LIBRARY: list[UgcAvatar] = [
    UgcAvatar(id="avatar-01", name="Sophia", image_url="", source="library"),
    UgcAvatar(id="avatar-02", name="Marcus", image_url="", source="library"),
    UgcAvatar(id="avatar-03", name="Aisha", image_url="", source="library"),
    UgcAvatar(id="avatar-04", name="Jake", image_url="", source="library"),
    UgcAvatar(id="avatar-05", name="Priya", image_url="", source="library"),
    UgcAvatar(id="avatar-06", name="Alex", image_url="", source="library"),
]


# ── Routes ──────────────────────────────────────────────────────────────


@router.get("/avatars")
async def list_avatars() -> UgcAvatarLibraryResponse:
    """Return the built-in avatar library."""
    return UgcAvatarLibraryResponse(avatars=AVATAR_LIBRARY)


@router.post("/avatars/upload")
async def upload_avatar(
    file: UploadFile = File(...),
) -> dict:
    """Upload a custom avatar portrait."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only image files are accepted.",
        )

    data = await file.read()
    storage = AssetStorage.from_settings()
    avatar_id = f"avatar-custom-{uuid.uuid4().hex[:8]}"
    file_name = file.filename or f"{avatar_id}.png"
    key = f"assets/ugc-avatars/{file_name}"

    await storage.save_asset(key=key, data=data, content_type=file.content_type)

    return {
        "avatar": UgcAvatar(
            id=avatar_id,
            name=file_name.rsplit(".", 1)[0],
            image_url=key,
            source="upload",
        ).model_dump()
    }


@router.post("/script/generate")
async def generate_script(
    body: UgcGenerateScriptRequest,
    db: AsyncSession = Depends(get_db),
) -> UgcGenerateScriptResponse:
    """Generate a UGC script from brand/product context."""
    brand = await _get_brand_or_404(body.brand_id, db)
    pipeline = _build_ugc_pipeline()

    script, used_fallback = await pipeline.generate_script(
        brand_name=brand.name or "",
        product_name=body.product_name or brand.name or "",
        product_description=body.product_description,
        target_audience=body.target_audience,
        key_benefit=body.key_benefit,
        tone=body.tone,
        cta_text=body.cta_text,
        target_duration_seconds=body.target_duration_seconds,
    )

    return UgcGenerateScriptResponse(script=script, used_fallback=used_fallback)


@router.post("/generate", status_code=status.HTTP_202_ACCEPTED)
async def generate_video(
    body: UgcGenerateVideoRequest,
    db: AsyncSession = Depends(get_db),
) -> UgcGenerateVideoResponse:
    """Kick off the full UGC video pipeline. Returns immediately with a job_id."""
    brand = await _get_brand_or_404(body.brand_id, db)
    pipeline = _build_ugc_pipeline()

    if not pipeline._fal.configured:
        if body.settings.render_mode == "talking_head":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="FAL_API_KEY is not configured.",
            )

    if body.settings.render_mode == "storyboard_action" and not pipeline.storyboard_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="VEO_PROJECT_ID, VEO_ACCESS_TOKEN, and VEO_GCS_BUCKET are required for storyboard mode.",
        )

    if not body.avatar.image_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Avatar has no image. Upload a custom avatar portrait first.",
        )

    job_id = uuid.uuid4().hex[:12]
    storage_prefix = f"assets/{body.brand_id}/ugc/{job_id}"

    job = UgcJobState(
        job_id=job_id,
        brand_id=body.brand_id,
        status="pending",
        steps=[UgcPipelineStep(step=name) for name in UgcPipelineService.STEP_NAMES],
        avatar=body.avatar,
        product_image_url=body.product_image_url,
        product_name=body.product_name,
        script=body.script,
        settings=body.settings,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    _job_store[job_id] = job

    def _on_update(updated_job: UgcJobState) -> None:
        _job_store[job_id] = updated_job

    # Run pipeline in background
    asyncio.create_task(
        pipeline.run_pipeline(
            job=job,
            request=body,
            brand_name=brand.name or "",
            storage_prefix=storage_prefix,
            on_update=_on_update,
        )
    )

    return UgcGenerateVideoResponse(job=job)


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str) -> UgcJobStatusResponse:
    """Poll the status of a running UGC generation job."""
    job = _job_store.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found.",
        )
    return UgcJobStatusResponse(job=job)
