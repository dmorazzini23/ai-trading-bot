from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from ai_trading.execution.engine import OrderManager
from ai_trading.oms.intent_store import IntentStore


pytest.importorskip("sqlalchemy")


def test_intent_store_uses_database_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db_path = tmp_path / "oms_from_database_url.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    store = IntentStore()
    record, created = store.create_intent(
        intent_id="intent-db-url-1",
        idempotency_key="db-url-key-1",
        symbol="AAPL",
        side="buy",
        quantity=3.0,
        status="PENDING_SUBMIT",
    )
    assert created is True
    assert record.intent_id == "intent-db-url-1"
    assert store.database_url.startswith("sqlite:///")
    assert db_path.exists()


def test_order_manager_live_requires_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.setenv("AI_TRADING_OMS_INTENT_STORE_IN_TESTS", "1")
    monkeypatch.setenv("AI_TRADING_OMS_INTENT_STORE_ENABLED", "1")
    monkeypatch.setenv("EXECUTION_MODE", "live")
    monkeypatch.setenv("AI_TRADING_OMS_INTENT_STORE_ALLOW_SQLITE_LIVE", "0")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(RuntimeError, match="DATABASE_URL is required"):
        OrderManager()


def test_order_manager_live_can_opt_into_sqlite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.setenv("AI_TRADING_OMS_INTENT_STORE_IN_TESTS", "1")
    monkeypatch.setenv("AI_TRADING_OMS_INTENT_STORE_ENABLED", "1")
    monkeypatch.setenv("EXECUTION_MODE", "live")
    monkeypatch.setenv("AI_TRADING_OMS_INTENT_STORE_ALLOW_SQLITE_LIVE", "1")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("AI_TRADING_OMS_INTENT_STORE_PATH", str(tmp_path / "oms_live_sqlite.db"))

    manager = OrderManager()
    assert manager._intent_store is not None


def test_migrate_oms_intent_store_script_round_trip(tmp_path: Path) -> None:
    source_db = tmp_path / "legacy_intents.db"
    target_db = tmp_path / "migrated_intents.db"

    source_store = IntentStore(path=str(source_db))
    seeded, created = source_store.create_intent(
        intent_id="intent-source-1",
        idempotency_key="source-key-1",
        symbol="MSFT",
        side="sell",
        quantity=2.0,
        status="SUBMITTED",
    )
    assert created is True
    source_store.mark_submitted(seeded.intent_id, "broker-source-1")
    source_store.record_fill(
        seeded.intent_id,
        fill_qty=2.0,
        fill_price=410.5,
        fee=0.1,
    )
    source_store.close_intent(seeded.intent_id, final_status="FILLED")

    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "migrate_oms_intent_store.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--source-sqlite",
            str(source_db),
            "--target-url",
            f"sqlite:///{target_db}",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(repo_root),
    )
    last_line = next(
        (line for line in reversed(proc.stdout.splitlines()) if line.strip()),
        "{}",
    )
    payload = json.loads(last_line)
    assert payload["migrated_intents"] >= 1
    assert payload["migrated_fills"] >= 1

    target_store = IntentStore(url=f"sqlite:///{target_db}")
    migrated = target_store.get_intent_by_key("source-key-1")
    assert migrated is not None
    assert migrated.broker_order_id == "broker-source-1"
    assert migrated.status == "FILLED"
    fills = target_store.list_fills(migrated.intent_id)
    assert len(fills) == 1
    assert fills[0].fill_price == pytest.approx(410.5)
