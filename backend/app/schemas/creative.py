"""Schemas for the creative studio flow."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.brand import BrandProfile


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
