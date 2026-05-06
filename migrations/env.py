"""Alembic migration environment for OMS persistence schema."""

from __future__ import annotations

from logging.config import fileConfig
import os
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = None
try:
    from ai_trading.oms.intent_store import _METADATA as _INTENT_METADATA
except Exception:
    _INTENT_METADATA = None

if _INTENT_METADATA is not None:
    target_metadata = _INTENT_METADATA


def _normalize_database_url(raw: str) -> str:
    value = str(raw).strip()
    if not value:
        return value
    if value.startswith("postgres://"):
        return f"postgresql+psycopg://{value[len('postgres://') :]}"
    if value.startswith("postgresql://") and "+" not in value.split("://", 1)[0]:
        return f"postgresql+psycopg://{value[len('postgresql://') :]}"
    return value


def _resolve_sqlalchemy_url() -> str:
    database_url = _normalize_database_url(os.getenv("DATABASE_URL", ""))
    if database_url:
        return database_url

    store_path = str(
        os.getenv("AI_TRADING_OMS_INTENT_STORE_PATH", "runtime/oms_intents.db")
    ).strip()
    if "://" in store_path:
        return _normalize_database_url(store_path)

    resolved_path = Path(store_path).expanduser()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{resolved_path}"


def run_migrations_offline() -> None:
    """Run migrations in offline mode."""
    url = _resolve_sqlalchemy_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in online mode."""
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = _resolve_sqlalchemy_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
