"""create brand tables

Revision ID: 20260325_0000
Revises:
Create Date: 2026-03-25 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "20260325_0000"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "brands",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("user_id", sa.String(length=255), nullable=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("website_url", sa.Text(), nullable=False),
        sa.Column("logo_url", sa.Text(), nullable=True),
        sa.Column("colors", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("fonts", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("voice", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("value_propositions", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("target_audience", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("products", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("industry", sa.String(length=255), nullable=True),
        sa.Column("raw_analysis", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("analysis_status", sa.String(length=20), server_default="pending", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_brands_user_id"), "brands", ["user_id"], unique=False)

    op.create_table(
        "brand_assets",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("brand_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("source_page", sa.Text(), nullable=True),
        sa.Column("stored_url", sa.Text(), nullable=True),
        sa.Column("file_name", sa.String(length=500), nullable=True),
        sa.Column("file_size", sa.Integer(), nullable=True),
        sa.Column("mime_type", sa.String(length=100), nullable=True),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("category", sa.String(length=50), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("tags", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("quality_score", sa.Integer(), nullable=True),
        sa.Column("is_usable", sa.Boolean(), server_default="true", nullable=False),
        sa.Column("alt_text", sa.Text(), nullable=True),
        sa.Column("context", sa.Text(), nullable=True),
        sa.Column("extraction_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["brand_id"], ["brands.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_brand_assets_brand_id"), "brand_assets", ["brand_id"], unique=False)
    op.create_index(op.f("ix_brand_assets_category"), "brand_assets", ["category"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_brand_assets_category"), table_name="brand_assets")
    op.drop_index(op.f("ix_brand_assets_brand_id"), table_name="brand_assets")
    op.drop_table("brand_assets")
    op.drop_index(op.f("ix_brands_user_id"), table_name="brands")
    op.drop_table("brands")
