"""Schemas for the UGC Studio pipeline."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.creative import VideoRenderRuntimeConfig


# ── Avatar ──────────────────────────────────────────────────────────────


class UgcAvatar(BaseModel):
    """An avatar portrait for UGC video generation."""

    id: str = ""
    name: str = ""
    image_url: str = ""
    source: Literal["library", "upload"] = "library"


# ── Script ──────────────────────────────────────────────────────────────


class UgcScriptSegment(BaseModel):
    """One segment of the UGC script."""

    role: Literal["hook", "body", "cta"]
    text: str
    duration_hint_seconds: float = 0.0


class UgcScript(BaseModel):
    """Three-part UGC script: hook, body, CTA."""

    segments: list[UgcScriptSegment] = Field(default_factory=list)
    full_text: str = ""
    estimated_duration_seconds: float = 0.0


# ── Settings ────────────────────────────────────────────────────────────


class UgcVideoSettings(BaseModel):
    """User-configurable settings for UGC video generation."""

    target_duration_seconds: int = 25
    aspect_ratio: Literal["9:16", "16:9", "1:1"] = "9:16"
    resolution: Literal["720p", "1080p"] = "720p"
    render_mode: Literal["talking_head", "storyboard_action"] = "storyboard_action"
    scenario: Literal[
        "product_demo",
        "closet",
        "bathroom",
        "bedroom",
        "kitchen",
        "desk",
        "car",
        "gym",
    ] = "product_demo"
    scene_count: int = 4
    tts_voice_name: str = "en-US-Studio-O"
    tts_speaking_rate: float = 1.0
    tts_pitch: float = 0.0
    include_music: bool = True
    music_prompt: str = "upbeat modern background music for advertisement"
    music_intensity: str = "low"
    include_captions: bool = True
    caption_style: Literal["minimal", "bold", "karaoke"] = "bold"
    broll_count: int = 2
    broll_duration_seconds: float = 3.0


# ── Pipeline Artifacts ──────────────────────────────────────────────────


class UgcArtifact(BaseModel):
    """A generated media artifact from the UGC pipeline."""

    kind: str  # composite_image, voiceover, talking_head, broll, music, final_video
    label: str
    stored_url: str | None = None
    mime_type: str = ""
    file_name: str | None = None


class UgcPipelineStep(BaseModel):
    """Status of a single pipeline step."""

    step: str  # script, composite, tts, talking_head, broll, compose
    status: Literal["pending", "running", "completed", "failed", "skipped"] = "pending"
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None


# ── Job State ───────────────────────────────────────────────────────────


class UgcJobState(BaseModel):
    """Full state of a UGC generation job."""

    job_id: str = ""
    brand_id: uuid.UUID | None = None
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    steps: list[UgcPipelineStep] = Field(default_factory=list)
    avatar: UgcAvatar | None = None
    product_image_url: str = ""
    product_name: str = ""
    script: UgcScript | None = None
    settings: UgcVideoSettings = Field(default_factory=UgcVideoSettings)
    artifacts: list[UgcArtifact] = Field(default_factory=list)
    final_video_url: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    error: str | None = None


# ── Request / Response ──────────────────────────────────────────────────


class UgcGenerateScriptRequest(BaseModel):
    """Request to generate a UGC script from brand context."""

    brand_id: uuid.UUID
    product_name: str = ""
    product_description: str = ""
    target_audience: str = ""
    key_benefit: str = ""
    tone: str = "conversational"
    cta_text: str = ""
    target_duration_seconds: int = 25


class UgcGenerateScriptResponse(BaseModel):
    """Generated UGC script."""

    script: UgcScript
    used_fallback: bool = False


class UgcGenerateVideoRequest(BaseModel):
    """Full request to kick off UGC video generation."""

    brand_id: uuid.UUID
    avatar: UgcAvatar
    product_image_url: str
    product_name: str = ""
    script: UgcScript
    settings: UgcVideoSettings = Field(default_factory=UgcVideoSettings)
    runtime: VideoRenderRuntimeConfig | None = None


class UgcGenerateVideoResponse(BaseModel):
    """Response after kicking off video generation."""

    job: UgcJobState


class UgcJobStatusResponse(BaseModel):
    """Polling response for a running job."""

    job: UgcJobState


class UgcAvatarLibraryResponse(BaseModel):
    """List of available avatars."""

    avatars: list[UgcAvatar] = Field(default_factory=list)
