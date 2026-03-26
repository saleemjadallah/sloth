"""Campaign ORM model."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import TYPE_CHECKING

from sqlalchemy import Date, DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.brand import Brand
    from app.models.creative_execution import CreativeExecution


class Campaign(Base):
    """Persisted campaign plan tied to a brand and optional executions."""

    __tablename__ = "campaigns"

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
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
        default="draft",
        server_default="draft",
    )
    objective: Mapped[str] = mapped_column(Text, nullable=False, default="", server_default="")
    audience_summary: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="",
        server_default="",
    )
    offer_summary: Mapped[str] = mapped_column(Text, nullable=False, default="", server_default="")
    primary_kpi: Mapped[str] = mapped_column(
        String(120),
        nullable=False,
        default="",
        server_default="",
    )
    budget_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    start_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    end_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    cadence_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    channels: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    messaging_pillars: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
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

    brand: Mapped["Brand"] = relationship(back_populates="campaigns")
    creative_executions: Mapped[list["CreativeExecution"]] = relationship(
        back_populates="campaign",
        lazy="selectin",
    )

    @property
    def linked_execution_count(self) -> int:
        return len(self.creative_executions) if self.creative_executions else 0

    @property
    def scheduled_execution_count(self) -> int:
        if not self.creative_executions:
            return 0
        return sum(1 for execution in self.creative_executions if execution.scheduled_for)

    @property
    def published_execution_count(self) -> int:
        if not self.creative_executions:
            return 0
        return sum(1 for execution in self.creative_executions if execution.published_at)

    def __repr__(self) -> str:
        return f"<Campaign id={self.id!s} name={self.name!r}>"
