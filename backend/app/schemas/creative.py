"""Schemas for the creative studio flow."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.brand import BrandAssetResponse, BrandProfile


class CreativeBrief(BaseModel):
    """High-level creative direction derived from a brand profile."""

    primary_goal: str = ""
    audience_focus: str = ""
    offer_summary: str = ""
    messaging_pillars: list[str] = Field(default_factory=list)
    tone_guardrails: list[str] = Field(default_factory=list)
    visual_direction: list[str] = Field(default_factory=list)
    recommended_formats: list[str] = Field(default_factory=list)


class StoryboardBeat(BaseModel):
    """Single storyboard beat for a concept."""

    step: str
    detail: str


class CreativeConcept(BaseModel):
    """One ad concept for a specific brand."""

    id: str
    name: str
    format: str
    angle: str
    hook: str
    primary_text: str
    cta: str
    why_it_will_work: str
    visual_direction: list[str] = Field(default_factory=list)
    asset_ids: list[str] = Field(default_factory=list)
    storyboard: list[StoryboardBeat] = Field(default_factory=list)


class CreativeStudioResponse(BaseModel):
    """Creative studio payload returned for a brand."""

    brand_id: uuid.UUID
    brand_name: str
    summary: str
    used_fallback: bool = False
    generated_at: datetime
    brief: CreativeBrief
    concepts: list[CreativeConcept] = Field(default_factory=list)


class CreativeExecutionRequest(BaseModel):
    """Selected brief and concept to expand into production outputs."""

    brief: CreativeBrief
    concept: CreativeConcept


class ChannelVariant(BaseModel):
    """Channel-specific version of the concept."""

    channel: str
    format: str
    headline: str
    primary_text: str
    cta: str


class DesignBrief(BaseModel):
    """Design guidance for static or motion creative production."""

    layout_direction: str
    asset_strategy: str
    copy_hierarchy: list[str] = Field(default_factory=list)
    visual_notes: list[str] = Field(default_factory=list)


class VideoBrief(BaseModel):
    """Video execution guidance for storyboarding and rendering."""

    concept: str
    opening_shot: str
    shot_list: list[str] = Field(default_factory=list)
    voiceover_script: str
    end_frame: str
    veo_prompt: str


class VideoScene(BaseModel):
    """One renderable scene derived from a video brief."""

    id: str
    title: str
    prompt: str
    duration_seconds: int = 8
    voiceover_text: str = ""
    reference_asset_ids: list[str] = Field(default_factory=list)
    enabled: bool = True
    sequence_index: int = 0
    concept_name: str = ""


class VideoRenderSettings(BaseModel):
    """Runtime settings for Veo/TTS/music generation."""

    render_strategy: Literal["scene_sequence", "daisy_chain"] = "scene_sequence"
    generation_mode: Literal[
        "auto",
        "prompt_only",
        "image_to_video",
        "reference_images",
        "first_last_frame",
    ] = "auto"
    model_id: str = "veo-3.1-generate-preview"
    aspect_ratio: Literal["9:16", "16:9"] = "9:16"
    resolution: str = "1080p"
    scene_duration_seconds: int = 8
    negative_prompt: str = ""
    seed: int | None = None
    generate_native_audio: bool = False
    create_voiceover: bool = True
    create_music: bool = False
    compose_final: bool = True
    stitch_scenes: bool = True
    tts_voice_name: str = "en-US-Studio-O"
    tts_speaking_rate: float = 1.0
    tts_pitch: float = 0.0
    tts_effects_profile_id: str = "headphone-class-device"
    music_prompt: str = ""
    music_intensity: str = "medium"
    music_mode: str = "track"
    music_format: str = "wav"
    music_bitrate: int = 320
    reference_asset_ids: list[str] = Field(default_factory=list)
    first_frame_asset_id: str | None = None
    last_frame_asset_id: str | None = None


class VideoRenderArtifact(BaseModel):
    """Stored media artifact emitted by the render pipeline."""

    kind: str
    label: str
    stored_url: str | None = None
    asset_id: str | None = None
    mime_type: str = ""
    file_name: str | None = None
    source_gcs_uri: str | None = None
    scene_id: str | None = None


class VideoRenderState(BaseModel):
    """Persisted render state attached to an execution pack."""

    status: Literal["idle", "running", "completed", "failed"] = "idle"
    provider: str = "vertex_veo"
    settings: VideoRenderSettings = Field(default_factory=VideoRenderSettings)
    scenes: list[VideoScene] = Field(default_factory=list)
    logs: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    scene_artifacts: list[VideoRenderArtifact] = Field(default_factory=list)
    stitched_artifact: VideoRenderArtifact | None = None
    voiceover_artifact: VideoRenderArtifact | None = None
    music_artifact: VideoRenderArtifact | None = None
    final_artifact: VideoRenderArtifact | None = None
    last_rendered_at: datetime | None = None


class CreativeExecutionResponse(BaseModel):
    """Expanded execution pack for a selected concept."""

    brand_id: uuid.UUID
    brand_name: str
    concept_id: str
    concept_name: str
    summary: str
    used_fallback: bool = False
    generated_at: datetime
    headlines: list[str] = Field(default_factory=list)
    primary_text_variants: list[str] = Field(default_factory=list)
    ctas: list[str] = Field(default_factory=list)
    channel_variants: list[ChannelVariant] = Field(default_factory=list)
    design_brief: DesignBrief
    video_brief: VideoBrief
    video_render: VideoRenderState = Field(default_factory=VideoRenderState)
    production_checklist: list[str] = Field(default_factory=list)


class SavedCreativeExecutionCreate(BaseModel):
    """Request body for persisting an execution pack."""

    brief: CreativeBrief
    concept: CreativeConcept
    execution: CreativeExecutionResponse
    delivery_mode: str = "late_dev"
    status: str = "draft"
    destination_label: str | None = None


class SavedCreativeExecutionSummary(BaseModel):
    """Lightweight saved execution item for listing."""

    id: uuid.UUID
    brand_id: uuid.UUID
    campaign_id: uuid.UUID | None = None
    concept_id: str
    concept_name: str
    summary: str
    delivery_mode: str
    status: str
    destination_label: str | None = None
    external_post_id: str | None = None
    external_post_url: str | None = None
    last_publish_error: str | None = None
    scheduled_for: datetime | None = None
    published_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class SavedCreativeExecutionResponse(BaseModel):
    """Persisted execution pack with reusable metadata."""

    id: uuid.UUID
    brand_id: uuid.UUID
    campaign_id: uuid.UUID | None = None
    concept_id: str
    concept_name: str
    summary: str
    delivery_mode: str
    status: str
    destination_label: str | None = None
    external_post_id: str | None = None
    external_post_url: str | None = None
    last_publish_error: str | None = None
    publishing_metadata: dict | None = None
    scheduled_for: datetime | None = None
    published_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    brief: CreativeBrief
    concept: CreativeConcept
    execution: CreativeExecutionResponse


class LateAccountProfile(BaseModel):
    id: str
    name: str
    slug: str | None = None


class LateAccountResponse(BaseModel):
    id: str
    platform: str
    username: str | None = None
    display_name: str | None = None
    profile_url: str | None = None
    is_active: bool = True
    profile: LateAccountProfile | None = None


class PublishSavedCreativeExecutionRequest(BaseModel):
    account_ids: list[str] = Field(default_factory=list)
    mode: Literal["draft", "publish_now", "schedule"] = "publish_now"
    timezone: str = "UTC"
    scheduled_for: datetime | None = None
    title: str | None = None
    content_override: str | None = None


class PublishSavedCreativeExecutionResponse(BaseModel):
    saved_execution: SavedCreativeExecutionResponse
    remote_post_id: str | None = None
    remote_post_status: str | None = None
    remote_post_url: str | None = None
    message: str = ""


class WorkspaceDeliveryState(BaseModel):
    delivery_mode: str = "late_dev"
    destination_label: str | None = None
    selected_late_profile_id: str | None = None
    publish_title: str | None = None
    content_override: str | None = None
    selected_late_account_ids: list[str] = Field(default_factory=list)
    publish_mode: Literal["draft", "publish_now", "schedule"] = "publish_now"
    scheduled_for: datetime | None = None


class BrandWorkspaceResponse(BaseModel):
    brand: BrandProfile
    studio: CreativeStudioResponse | None = None
    selected_concept_id: str | None = None
    selected_asset_ids: list[str] = Field(default_factory=list)
    execution: CreativeExecutionResponse | None = None
    saved_execution_id: uuid.UUID | None = None
    delivery: WorkspaceDeliveryState = Field(default_factory=WorkspaceDeliveryState)
    saved_executions: list[SavedCreativeExecutionSummary] = Field(default_factory=list)


class BrandWorkspaceUpdate(BaseModel):
    studio: CreativeStudioResponse | None = None
    selected_concept_id: str | None = None
    selected_asset_ids: list[str] = Field(default_factory=list)
    execution: CreativeExecutionResponse | None = None
    saved_execution_id: uuid.UUID | None = None
    delivery: WorkspaceDeliveryState = Field(default_factory=WorkspaceDeliveryState)


class CreativeVideoRenderRequest(BaseModel):
    """Request body for in-app video rendering."""

    execution: CreativeExecutionResponse
    selected_asset_ids: list[uuid.UUID] = Field(default_factory=list)
    regenerate_scenes: bool = False
    runtime: "VideoRenderRuntimeConfig"


class CreativeVideoRenderResponse(BaseModel):
    """Updated execution plus any created/persisted assets."""

    execution: CreativeExecutionResponse
    created_assets: list[BrandAssetResponse] = Field(default_factory=list)


class VideoRenderRuntimeConfig(BaseModel):
    """Short-lived runtime credentials provided by the frontend."""

    project_id: str
    access_token: str
    gcs_bucket: str
    location: str = "us-central1"
