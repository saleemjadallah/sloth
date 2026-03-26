"""Brand ORM model."""

from __future__ import annotations

import uuid
from datetime import datetime

from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.brand_asset import BrandAsset
    from app.models.campaign import Campaign
    from app.models.creative_execution import CreativeExecution


class Brand(Base):
    """Represents a brand extracted from a website analysis."""

    __tablename__ = "brands"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid(),
    )
    user_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
        comment="Will be populated once auth is wired up.",
    )

    # ── Core fields ─────────────────────────────────────────────────────
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    website_url: Mapped[str] = mapped_column(Text, nullable=False)
    logo_url: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ── Brand identity (JSONB) ──────────────────────────────────────────
    colors: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    fonts: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    voice: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    value_propositions: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    target_audience: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    products: Mapped[list | None] = mapped_column(JSONB, nullable=True)

    # ── Metadata ────────────────────────────────────────────────────────
    industry: Mapped[str | None] = mapped_column(String(255), nullable=True)
    raw_analysis: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    analysis_status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="pending",
        server_default="pending",
    )
    workspace_studio: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    workspace_concept_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    workspace_selected_concept_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    workspace_selected_asset_ids: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    workspace_execution: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    workspace_delivery: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    workspace_saved_execution_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )

    # ── Timestamps ──────────────────────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    # ── Relationships ────────────────────────────────────────────────────
    assets: Mapped[list["BrandAsset"]] = relationship(
        back_populates="brand",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    creative_executions: Mapped[list["CreativeExecution"]] = relationship(
        back_populates="brand",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    campaigns: Mapped[list["Campaign"]] = relationship(
        back_populates="brand",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    # ── Computed properties (for Pydantic serialization) ──────────────────
    @property
    def asset_count(self) -> int:
        return len(self.assets) if self.assets else 0

    @property
    def usable_asset_count(self) -> int:
        if not self.assets:
            return 0
        return sum(1 for a in self.assets if a.is_usable)

    @property
    def saved_execution_count(self) -> int:
        return len(self.creative_executions) if self.creative_executions else 0

    @property
    def published_execution_count(self) -> int:
        if not self.creative_executions:
            return 0
        return sum(
            1
            for execution in self.creative_executions
            if execution.external_post_id or execution.published_at or execution.scheduled_for
        )

    @property
    def active_execution_status(self) -> str | None:
        latest = self._latest_execution
        return latest.status if latest else None

    @property
    def active_execution_updated_at(self) -> datetime | None:
        latest = self._latest_execution
        return latest.updated_at if latest else None

    @property
    def active_execution_last_error(self) -> str | None:
        latest = self._latest_execution
        return latest.last_publish_error if latest else None

    @property
    def _latest_execution(self) -> "CreativeExecution" | None:
        if not self.creative_executions:
            return None
        return max(
            self.creative_executions,
            key=lambda execution: execution.updated_at or execution.created_at,
        )

    def __repr__(self) -> str:
        return f"<Brand id={self.id!s} name={self.name!r}>"
