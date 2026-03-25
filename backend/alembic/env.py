"""Alembic environment – async migration runner."""

from __future__ import annotations

import asyncio
import os

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

# Import the declarative Base so Alembic can see table metadata, and
# import all models so they register on Base.metadata.
from app.database import Base
from app.models import *  # noqa: F401, F403

# Alembic Config object (provides access to alembic.ini values).
config = context.config

# Override the URL with an env var when available (e.g. in CI / production).
database_url = os.getenv(
    "DATABASE_URL",
    config.get_main_option("sqlalchemy.url"),
)
if database_url:
    # Ensure the URL uses the asyncpg driver for async migrations.
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    config.set_main_option("sqlalchemy.url", database_url)

# Target metadata for autogenerate.
target_metadata = Base.metadata


# ── Offline mode (generates SQL scripts without a live DB) ──────────────

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


# ── Online mode (connects to the database) ─────────────────────────────

def do_run_migrations(connection) -> None:  # noqa: ANN001
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine and run migrations inside a connection."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


# ── Entrypoint ──────────────────────────────────────────────────────────

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
