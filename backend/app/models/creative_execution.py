"""Saved creative execution packs for export and publishing workflows."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.campaign import Campaign
    from app.models.brand import Brand


class CreativeExecution(Base):
    """Persisted concept execution output for reuse, export, and publishing."""

    __tablename__ = "creative_executions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid(),
    )
    brand_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("brands.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    campaign_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("campaigns.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    concept_id: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    concept_name: Mapped[str] = mapped_column(String(255), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="", server_default="")
    delivery_mode: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
        default="late_dev",
        server_default="late_dev",
        comment="late_dev, export_only, or hybrid",
    )
    status: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
        default="draft",
        server_default="draft",
        comment="draft, approved, queued, published",
    )
    destination_label: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="Optional publishing target label, such as late.dev workspace/campaign",
    )
    external_post_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="Late/Zernio post ID for published or scheduled content",
    )
    external_post_url: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Public platform URL if available after publish",
    )
    last_publish_error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Last publish error from Late/Zernio if a publish attempt failed",
    )
    brief: Mapped[dict] = mapped_column(JSONB, nullable=False)
    concept: Mapped[dict] = mapped_column(JSONB, nullable=False)
    execution: Mapped[dict] = mapped_column(JSONB, nullable=False)
    publishing_metadata: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

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
    scheduled_for: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    brand: Mapped["Brand"] = relationship(back_populates="creative_executions")
    campaign: Mapped["Campaign | None"] = relationship(back_populates="creative_executions")

    def __repr__(self) -> str:
        return f"<CreativeExecution id={self.id!s} concept_name={self.concept_name!r}>"
