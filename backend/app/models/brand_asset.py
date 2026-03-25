"""Brand asset ORM model — images and media extracted from a brand's website."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class BrandAsset(Base):
    """An image or media file extracted from a brand's website."""

    __tablename__ = "brand_assets"

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

    # ── Source info ───────────────────────────────────────────────────────
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    source_page: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="Page URL where this asset was found"
    )

    # ── Storage ──────────────────────────────────────────────────────────
    stored_url: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="R2/local URL after download"
    )
    file_name: Mapped[str | None] = mapped_column(String(500), nullable=True)
    file_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mime_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # ── Classification (LLM-powered) ─────────────────────────────────────
    category: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        index=True,
        comment="product, hero, lifestyle, logo, icon, team, testimonial, banner, screenshot, other",
    )
    description: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="LLM-generated description of the asset"
    )
    tags: Mapped[list | None] = mapped_column(
        JSONB, nullable=True, comment="Searchable tags for the asset"
    )
    quality_score: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="1-10 score for ad-creation suitability",
    )
    is_usable: Mapped[bool] = mapped_column(
        default=True,
        server_default="true",
        comment="Whether this asset is suitable for ad creation",
    )

    # ── Metadata ─────────────────────────────────────────────────────────
    alt_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    context: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="Surrounding text/context from the page"
    )
    extraction_metadata: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True, comment="Raw extraction data (position, css class, etc.)"
    )

    # ── Timestamps ───────────────────────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # ── Relationships ────────────────────────────────────────────────────
    brand: Mapped["Brand"] = relationship(back_populates="assets")  # noqa: F821

    def __repr__(self) -> str:
        return f"<BrandAsset id={self.id!s} category={self.category!r}>"
