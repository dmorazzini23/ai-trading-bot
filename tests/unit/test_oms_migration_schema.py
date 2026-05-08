from __future__ import annotations

import sqlite3
import logging
from pathlib import Path

import pytest


pytest.importorskip("alembic")


def test_alembic_head_creates_unique_oms_intent_sequence_index(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from alembic import command
    from alembic.config import Config

    db_path = tmp_path / "oms_migration_schema.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("AI_TRADING_OMS_INTENT_STORE_PATH", str(db_path))

    config = Config(str(Path.cwd() / "alembic.ini"))
    command.upgrade(config, "head")

    assert logging.getLogger("ai_trading.risk.engine").disabled is False

    with sqlite3.connect(db_path) as conn:
        revision = conn.execute("SELECT version_num FROM alembic_version LIMIT 1").fetchone()
        columns = {
            row[1]: row
            for row in conn.execute("PRAGMA table_info(oms_events)").fetchall()
        }
        indexes = {
            row[1]: row
            for row in conn.execute("PRAGMA index_list(oms_events)").fetchall()
        }

    assert revision == ("20260506_0001",)
    assert {"intent_id", "sequence_no", "event_source", "idempotency_key"} <= set(columns)
    assert indexes["uq_oms_events_intent_sequence"][2] == 1


def test_oms_intent_sequence_migration_dedupes_before_unique_index(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from alembic import command
    from alembic.config import Config

    db_path = tmp_path / "oms_migration_dedupe.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("AI_TRADING_OMS_INTENT_STORE_PATH", str(db_path))

    config = Config(str(Path.cwd() / "alembic.ini"))
    command.upgrade(config, "20260414_0001")
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO oms_events (
                event_uuid,
                intent_id,
                event_type,
                event_ts,
                event_source,
                idempotency_key,
                sequence_no,
                payload_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "event-1",
                    "intent-1",
                    "INTENT_CREATED",
                    "2026-05-06T00:00:00Z",
                    "unit_test",
                    "key-1",
                    1,
                    "{}",
                    "2026-05-06T00:00:00Z",
                ),
                (
                    "event-2",
                    "intent-1",
                    "ORDER_SUBMITTED",
                    "2026-05-06T00:00:01Z",
                    "unit_test",
                    "key-2",
                    1,
                    "{}",
                    "2026-05-06T00:00:01Z",
                ),
                (
                    "event-empty-1",
                    "",
                    "RECONCILE_UPDATE",
                    "2026-05-06T00:00:02Z",
                    "unit_test",
                    "key-empty-1",
                    0,
                    "{}",
                    "2026-05-06T00:00:02Z",
                ),
                (
                    "event-empty-2",
                    "",
                    "RECONCILE_UPDATE",
                    "2026-05-06T00:00:03Z",
                    "unit_test",
                    "key-empty-2",
                    0,
                    "{}",
                    "2026-05-06T00:00:03Z",
                ),
            ],
        )
        conn.execute(
            """
            CREATE TRIGGER trg_oms_events_no_update
            BEFORE UPDATE ON oms_events
            BEGIN
              SELECT RAISE(ABORT, 'oms_events is append-only');
            END
            """
        )
        conn.execute(
            """
            CREATE TRIGGER trg_oms_events_no_delete
            BEFORE DELETE ON oms_events
            BEGIN
              SELECT RAISE(ABORT, 'oms_events is append-only');
            END
            """
        )

    command.upgrade(config, "head")

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT event_uuid, sequence_no FROM oms_events ORDER BY event_id"
        ).fetchall()
        duplicate_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT intent_id, sequence_no, COUNT(*) AS row_count
                FROM oms_events
                WHERE intent_id IS NOT NULL
                GROUP BY intent_id, sequence_no
                HAVING row_count > 1
            )
            """
        ).fetchone()
        indexes = {
            row[1]: row
            for row in conn.execute("PRAGMA index_list(oms_events)").fetchall()
        }
        triggers = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name='oms_events'"
            ).fetchall()
        }
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute(
                "UPDATE oms_events SET sequence_no = sequence_no WHERE event_uuid='event-1'"
            )

    assert rows == [
        ("event-1", 1),
        ("event-2", 2),
        ("event-empty-1", 0),
        ("event-empty-2", 1),
    ]
    assert duplicate_count == (0,)
    assert indexes["uq_oms_events_intent_sequence"][2] == 1
    assert {"trg_oms_events_no_update", "trg_oms_events_no_delete"} <= triggers
