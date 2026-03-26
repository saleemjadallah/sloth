"""Schemas for campaign planning endpoints."""

from __future__ import annotations

import uuid
from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.creative import SavedCreativeExecutionSummary


class CampaignBrandSummary(BaseModel):
    """Small nested brand summary for campaign responses."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    website_url: str
    logo_url: str | None = None
    industry: str | None = None


class CampaignUpsert(BaseModel):
    """Body for creating or updating a campaign."""

    brand_id: uuid.UUID
    name: str = Field(min_length=2, max_length=255)
    status: str = Field(default="draft", max_length=40)
    objective: str = Field(default="", max_length=4_000)
    audience_summary: str = Field(default="", max_length=4_000)
    offer_summary: str = Field(default="", max_length=4_000)
    primary_kpi: str = Field(default="", max_length=120)
    budget_summary: str | None = Field(default=None, max_length=2_000)
    start_date: date | None = None
    end_date: date | None = None
    cadence_summary: str | None = Field(default=None, max_length=4_000)
    channels: list[str] = Field(default_factory=list)
    messaging_pillars: list[str] = Field(default_factory=list)
    notes: str | None = Field(default=None, max_length=6_000)
    execution_ids: list[uuid.UUID] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_dates(self) -> "CampaignUpsert":
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date must be on or before end_date")
        self.channels = [item.strip() for item in self.channels if item.strip()]
        self.messaging_pillars = [
            item.strip() for item in self.messaging_pillars if item.strip()
        ]
        return self


class CampaignSummary(BaseModel):
    """Lightweight campaign list item."""

    id: uuid.UUID
    brand_id: uuid.UUID
    brand_name: str
    brand_logo_url: str | None = None
    name: str
    status: str
    objective: str
    primary_kpi: str
    start_date: date | None = None
    end_date: date | None = None
    channels: list[str] = Field(default_factory=list)
    linked_execution_count: int = 0
    scheduled_execution_count: int = 0
    published_execution_count: int = 0
    created_at: datetime
    updated_at: datetime


class CampaignResponse(CampaignSummary):
    """Full campaign payload with brand and linked executions."""

    audience_summary: str = ""
    offer_summary: str = ""
    budget_summary: str | None = None
    cadence_summary: str | None = None
    messaging_pillars: list[str] = Field(default_factory=list)
    notes: str | None = None
    brand: CampaignBrandSummary
    executions: list[SavedCreativeExecutionSummary] = Field(default_factory=list)
