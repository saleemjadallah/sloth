"""create creative executions

Revision ID: 20260325_0001
Revises:
Create Date: 2026-03-25 00:01:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "20260325_0001"
down_revision = "20260325_0000"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "creative_executions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("brand_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("concept_id", sa.String(length=120), nullable=False),
        sa.Column("concept_name", sa.String(length=255), nullable=False),
        sa.Column("summary", sa.Text(), server_default="", nullable=False),
        sa.Column("delivery_mode", sa.String(length=40), server_default="late_dev", nullable=False),
        sa.Column("status", sa.String(length=40), server_default="draft", nullable=False),
        sa.Column("destination_label", sa.String(length=255), nullable=True),
        sa.Column("external_post_id", sa.String(length=255), nullable=True),
        sa.Column("external_post_url", sa.Text(), nullable=True),
        sa.Column("last_publish_error", sa.Text(), nullable=True),
        sa.Column("brief", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("concept", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("execution", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("publishing_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("scheduled_for", sa.DateTime(timezone=True), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["brand_id"], ["brands.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_creative_executions_brand_id"),
        "creative_executions",
        ["brand_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_creative_executions_concept_id"),
        "creative_executions",
        ["concept_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_creative_executions_concept_id"), table_name="creative_executions")
    op.drop_index(op.f("ix_creative_executions_brand_id"), table_name="creative_executions")
    op.drop_table("creative_executions")
