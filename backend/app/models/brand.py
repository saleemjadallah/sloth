"""Brand ORM model."""

from __future__ import annotations

import uuid
from datetime import datetime

from typing import TYPE_CHECKING

from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.brand_asset import BrandAsset


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

    # ── Computed properties (for Pydantic serialization) ──────────────────
    @property
    def asset_count(self) -> int:
        return len(self.assets) if self.assets else 0

    @property
    def usable_asset_count(self) -> int:
        if not self.assets:
            return 0
        return sum(1 for a in self.assets if a.is_usable)

    def __repr__(self) -> str:
        return f"<Brand id={self.id!s} name={self.name!r}>"
