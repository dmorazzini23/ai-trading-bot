from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_trading.tools.oms_lifecycle_parity_replay import (
    main,
    replay_lifecycle_parity,
)


pytest.importorskip("sqlalchemy")


def _fixture_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "oms_lifecycle_parity_fixture.json"


def test_replay_lifecycle_parity_reports_no_mismatches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "oms_lifecycle_parity_replay.db"
    monkeypatch.setenv("AI_TRADING_OMS_EVENT_DUAL_WRITE_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_OMS_EVENT_JSONL_ENABLED", "0")

    payload = replay_lifecycle_parity(
        fixture_path=str(_fixture_path()),
        database_url=f"sqlite:///{db_path}",
        intent_store_path=str(db_path),
    )
    assert payload["ok"] is True
    assert int(payload["mismatch_count"]) == 0
    assert int(payload["scenario_count"]) == 4
    comparisons = payload["comparisons"]
    assert isinstance(comparisons, list)
    assert all(bool(item.get("parity_ok")) for item in comparisons)
    assert str(payload["run_id"]).startswith("replay-")


def test_replay_lifecycle_parity_defaults_to_temp_database(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_db = tmp_path / "runtime_should_not_be_used.db"
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("AI_TRADING_OMS_INTENT_STORE_PATH", str(runtime_db))
    monkeypatch.setenv("AI_TRADING_OMS_EVENT_DUAL_WRITE_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_OMS_EVENT_JSONL_ENABLED", "0")

    payload = replay_lifecycle_parity(fixture_path=str(_fixture_path()))

    assert payload["ok"] is True
    assert payload["intent_store_path"] != str(runtime_db)
    assert "oms-lifecycle-parity-" in payload["intent_store_path"]
    assert not runtime_db.exists()


def test_replay_lifecycle_parity_cli_main_returns_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path = tmp_path / "oms_lifecycle_parity_replay_cli.db"
    monkeypatch.setenv("AI_TRADING_OMS_EVENT_DUAL_WRITE_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_OMS_EVENT_JSONL_ENABLED", "0")

    exit_code = main(
        [
            "--fixture",
            str(_fixture_path()),
            "--database-url",
            f"sqlite:///{db_path}",
            "--intent-store-path",
            str(db_path),
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload.get("ok") is True
    assert int(payload.get("mismatch_count", -1)) == 0
