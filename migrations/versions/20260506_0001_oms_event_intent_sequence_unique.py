"""Enforce unique OMS event sequence numbers per intent.

Revision ID: 20260506_0001
Revises: 20260414_0001
Create Date: 2026-05-06
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260506_0001"
down_revision = "20260414_0001"
branch_labels = None
depends_on = None


_UNIQUE_INDEX_NAME = "uq_oms_events_intent_sequence"
_OMS_APPEND_ONLY_TRIGGER_NAMES = (
    "trg_oms_events_no_update",
    "trg_oms_events_no_delete",
)


def _table_exists(table_name: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return bool(inspector.has_table(table_name))


def _index_exists(table_name: str, index_name: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    try:
        return any(index.get("name") == index_name for index in inspector.get_indexes(table_name))
    except sa.exc.SQLAlchemyError:
        return False


def _dedupe_intent_sequences() -> None:
    op.execute(
        sa.text(
            """
            WITH ranked AS (
                SELECT
                    event_id,
                    intent_id,
                    sequence_no,
                    ROW_NUMBER() OVER (
                        PARTITION BY intent_id, sequence_no
                        ORDER BY event_id
                    ) AS duplicate_rank,
                    MAX(sequence_no) OVER (
                        PARTITION BY intent_id
                    ) AS max_sequence_no
                FROM oms_events
                WHERE intent_id IS NOT NULL
            ),
            duplicate_rows AS (
                SELECT
                    event_id,
                    max_sequence_no + ROW_NUMBER() OVER (
                        PARTITION BY intent_id
                        ORDER BY sequence_no, event_id
                    ) AS replacement_sequence_no
                FROM ranked
                WHERE duplicate_rank > 1
            )
            UPDATE oms_events
            SET sequence_no = (
                SELECT replacement_sequence_no
                FROM duplicate_rows
                WHERE duplicate_rows.event_id = oms_events.event_id
            )
            WHERE event_id IN (SELECT event_id FROM duplicate_rows)
            """
        )
    )


def _drop_oms_append_only_guards() -> None:
    bind = op.get_bind()
    dialect_name = str(getattr(getattr(bind, "dialect", None), "name", "") or "")
    for trigger_name in _OMS_APPEND_ONLY_TRIGGER_NAMES:
        if dialect_name.startswith("postgres"):
            op.execute(sa.text(f"DROP TRIGGER IF EXISTS {trigger_name} ON oms_events"))
        else:
            op.execute(sa.text(f"DROP TRIGGER IF EXISTS {trigger_name}"))


def _restore_oms_append_only_guards() -> None:
    bind = op.get_bind()
    dialect_name = str(getattr(getattr(bind, "dialect", None), "name", "") or "")
    if dialect_name.startswith("postgres"):
        op.execute(
            sa.text(
                """
                CREATE OR REPLACE FUNCTION ai_trading_prevent_append_only_mutation()
                RETURNS trigger
                LANGUAGE plpgsql
                AS $$
                BEGIN
                  RAISE EXCEPTION '% is append-only', TG_TABLE_NAME;
                END;
                $$;
                """
            )
        )
        op.execute(
            sa.text(
                """
                CREATE TRIGGER trg_oms_events_no_update
                BEFORE UPDATE ON oms_events
                FOR EACH ROW
                EXECUTE FUNCTION ai_trading_prevent_append_only_mutation()
                """
            )
        )
        op.execute(
            sa.text(
                """
                CREATE TRIGGER trg_oms_events_no_delete
                BEFORE DELETE ON oms_events
                FOR EACH ROW
                EXECUTE FUNCTION ai_trading_prevent_append_only_mutation()
                """
            )
        )
        return
    op.execute(
        sa.text(
            """
            CREATE TRIGGER IF NOT EXISTS trg_oms_events_no_update
            BEFORE UPDATE ON oms_events
            BEGIN
              SELECT RAISE(ABORT, 'oms_events is append-only');
            END
            """
        )
    )
    op.execute(
        sa.text(
            """
            CREATE TRIGGER IF NOT EXISTS trg_oms_events_no_delete
            BEFORE DELETE ON oms_events
            BEGIN
              SELECT RAISE(ABORT, 'oms_events is append-only');
            END
            """
        )
    )


def upgrade() -> None:
    if not _table_exists("oms_events"):
        return
    _drop_oms_append_only_guards()
    try:
        _dedupe_intent_sequences()
    finally:
        _restore_oms_append_only_guards()
    if not _index_exists("oms_events", _UNIQUE_INDEX_NAME):
        op.create_index(
            _UNIQUE_INDEX_NAME,
            "oms_events",
            ["intent_id", "sequence_no"],
            unique=True,
        )


def downgrade() -> None:
    if _table_exists("oms_events") and _index_exists("oms_events", _UNIQUE_INDEX_NAME):
        op.drop_index(_UNIQUE_INDEX_NAME, table_name="oms_events")
