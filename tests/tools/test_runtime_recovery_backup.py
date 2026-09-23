from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from ai_trading.execution.engine import OrderManager
from ai_trading.oms.intent_store import IntentStore
from ai_trading.tools.runtime_recovery_backup import (
    create_backup,
    restore_to_directory,
    verify_backup,
)

pytest.importorskip("sqlalchemy")


def _seed(data_dir: Path) -> IntentStore:
    runtime_dir = data_dir / "runtime"
    models_dir = data_dir / "models"
    runtime_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)
    store = IntentStore(path=str(runtime_dir / "oms_intents.db"))
    intent, created = store.create_intent(
        intent_id="client-recovery-1",
        idempotency_key="recovery-key-1",
        symbol="AAPL",
        side="buy",
        quantity=2.0,
        status="PENDING_SUBMIT",
    )
    assert created
    assert store.claim_for_submit(intent.intent_id)
    with sqlite3.connect(runtime_dir / "pretrade_rate_limiter.db") as connection:
        connection.execute("CREATE TABLE limits (name TEXT PRIMARY KEY, used INTEGER)")
        connection.execute("INSERT INTO limits VALUES ('daily', 1)")
    (runtime_dir / "run_manifest.json").write_text(
        json.dumps({"git_commit_hash": "synthetic", "resolved_config_hash": "test"}),
        encoding="utf-8",
    )
    (runtime_dir / "release_spec.json").write_text(
        json.dumps({"schema_version": 1, "tested_commit_sha": "synthetic"}),
        encoding="utf-8",
    )
    (runtime_dir / "release_identity_preflight.json").write_text(
        json.dumps({"status": "pass"}), encoding="utf-8",
    )
    (runtime_dir / "live_canary_state_latest.json").write_text(
        json.dumps({"entry_attempts": 1}), encoding="utf-8"
    )
    (models_dir / "registry_index.json").write_text("{}", encoding="utf-8")
    (models_dir / "trained_model.pkl").write_bytes(b"synthetic-model")
    (models_dir / "credentials.json").write_text("secret", encoding="utf-8")
    (data_dir / ".env").write_text("ALPACA_SECRET_KEY=never-back-up", encoding="utf-8")
    return store


def test_bundle_restores_consistent_oms_and_reconciles_broker_truth(tmp_path: Path) -> None:
    data_dir = tmp_path / "source"
    _seed(data_dir)
    bundle = create_backup(data_dir, destination=tmp_path / "backups")

    manifest = verify_backup(bundle)
    names = {entry["path"] for entry in manifest["entries"]}
    assert "runtime/oms_intents.db" in names
    assert "runtime/pretrade_rate_limiter.db" in names
    assert "runtime/release_spec.json" in names
    assert "runtime/release_identity_preflight.json" in names
    assert "models/registry_index.json" in names
    assert "models/trained_model.pkl" in names
    assert not any("credential" in name or ".env" in name for name in names)
    assert manifest["credential_files_included"] is False

    restored_root = tmp_path / "restored"
    restore_to_directory(bundle, restored_root)
    assert json.loads((restored_root / "manifest.json").read_text())["schema_version"] == 1
    assert bundle.stat().st_mode & 0o777 == 0o600
    restored_store = IntentStore(path=str(restored_root / "runtime" / "oms_intents.db"))
    intent = restored_store.get_intent("client-recovery-1")
    assert intent is not None
    assert intent.status == "SUBMITTING"
    assert not restored_store.claim_for_submit(intent.intent_id, stale_after_seconds=1)
    manager = OrderManager()
    manager.configure_intent_store(restored_store)
    summary = manager.reconcile_open_intents(
        broker_orders=[],
        get_order_by_client_order_id_fn=lambda client_id: {
            "id": "broker-recovery-1",
            "client_order_id": client_id,
            "status": "accepted",
        },
    )
    recovered = restored_store.get_intent(intent.intent_id)
    assert recovered is not None
    assert recovered.status == "SUBMITTED"
    assert recovered.broker_order_id == "broker-recovery-1"
    assert summary["marked_submitted"] == 1


def test_bundle_retention_and_corruption_detection(tmp_path: Path) -> None:
    data_dir = tmp_path / "source"
    _seed(data_dir)
    backup_dir = tmp_path / "backups"
    first = create_backup(data_dir, destination=backup_dir, max_snapshots=1)
    second = create_backup(data_dir, destination=backup_dir, max_snapshots=1)
    assert first != second
    assert not first.exists()
    assert second.exists()
    with pytest.raises(FileExistsError):
        restore_to_directory(second, data_dir)
    corrupted = tmp_path / "corrupted.gz"
    corrupted.write_bytes(second.read_bytes()[:-64])
    with pytest.raises((OSError, RuntimeError, EOFError)):
        verify_backup(corrupted)


def test_backup_failure_records_status_without_source_details(tmp_path: Path) -> None:
    data_dir = tmp_path / "source"
    (data_dir / "runtime").mkdir(parents=True)
    (data_dir / "models").mkdir()
    with pytest.raises(RuntimeError, match="runtime_databases_missing"):
        create_backup(data_dir, destination=tmp_path / "backups")
    status = json.loads((data_dir / "runtime" / "recovery_backup_latest.json").read_text())
    assert status["status"] == "failed"
    assert status["reason"] == "RuntimeError"
    assert status["bundle"] is None
