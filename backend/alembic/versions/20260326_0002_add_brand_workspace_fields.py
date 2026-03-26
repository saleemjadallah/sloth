"""add brand workspace fields

Revision ID: 20260326_0002
Revises: 20260325_0001
Create Date: 2026-03-26 00:02:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "20260326_0002"
down_revision = "20260325_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "brands",
        sa.Column("workspace_studio", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column(
        "brands",
        sa.Column("workspace_concept_count", sa.Integer(), nullable=True),
    )
    op.add_column(
        "brands",
        sa.Column("workspace_selected_concept_id", sa.String(length=120), nullable=True),
    )
    op.add_column(
        "brands",
        sa.Column(
            "workspace_selected_asset_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )
    op.add_column(
        "brands",
        sa.Column("workspace_execution", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column(
        "brands",
        sa.Column("workspace_delivery", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column(
        "brands",
        sa.Column("workspace_saved_execution_id", postgresql.UUID(as_uuid=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("brands", "workspace_saved_execution_id")
    op.drop_column("brands", "workspace_delivery")
    op.drop_column("brands", "workspace_execution")
    op.drop_column("brands", "workspace_selected_asset_ids")
    op.drop_column("brands", "workspace_selected_concept_id")
    op.drop_column("brands", "workspace_concept_count")
    op.drop_column("brands", "workspace_studio")
