"""Schemas for the creative studio flow."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


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
