"""create campaigns

Revision ID: 20260326_0003
Revises: 20260326_0002
Create Date: 2026-03-26 00:03:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "20260326_0003"
down_revision = "20260326_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "campaigns",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("brand_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=40), server_default="draft", nullable=False),
        sa.Column("objective", sa.Text(), server_default="", nullable=False),
        sa.Column("audience_summary", sa.Text(), server_default="", nullable=False),
        sa.Column("offer_summary", sa.Text(), server_default="", nullable=False),
        sa.Column("primary_kpi", sa.String(length=120), server_default="", nullable=False),
        sa.Column("budget_summary", sa.Text(), nullable=True),
        sa.Column("start_date", sa.Date(), nullable=True),
        sa.Column("end_date", sa.Date(), nullable=True),
        sa.Column("cadence_summary", sa.Text(), nullable=True),
        sa.Column("channels", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("messaging_pillars", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["brand_id"], ["brands.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_campaigns_brand_id"), "campaigns", ["brand_id"], unique=False)

    op.add_column(
        "creative_executions",
        sa.Column("campaign_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_index(
        op.f("ix_creative_executions_campaign_id"),
        "creative_executions",
        ["campaign_id"],
        unique=False,
    )
    op.create_foreign_key(
        "fk_creative_executions_campaign_id_campaigns",
        "creative_executions",
        "campaigns",
        ["campaign_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_creative_executions_campaign_id_campaigns",
        "creative_executions",
        type_="foreignkey",
    )
    op.drop_index(op.f("ix_creative_executions_campaign_id"), table_name="creative_executions")
    op.drop_column("creative_executions", "campaign_id")

    op.drop_index(op.f("ix_campaigns_brand_id"), table_name="campaigns")
    op.drop_table("campaigns")
