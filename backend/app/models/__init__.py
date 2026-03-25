"""ORM models – import every model here so Alembic can detect them."""

from app.models.brand import Brand  # noqa: F401
from app.models.brand_asset import BrandAsset  # noqa: F401

__all__ = ["Brand", "BrandAsset"]
