from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ai_trading.core import bot_engine


class _DummyDecisionRecord:
    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": "AAPL",
            "config_snapshot": {
                "api_key": "secret-value",
                "nested": {"token": "secret-token"},
                "safe": "ok",
            },
        }


class _TimestampDecisionRecord:
    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": "AAPL",
            "bar_ts": datetime.now(UTC),
            "metrics": {"observed_at": datetime.now(UTC)},
        }


def _read_single_json(path: Path) -> dict[str, object]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return json.loads(lines[-1])


def test_write_decision_record_writes_secondary_snapshot_and_redacts(
    tmp_path: Path, monkeypatch
) -> None:
    primary = tmp_path / "decision.jsonl"
    snapshot = tmp_path / "snapshots.jsonl"
    monkeypatch.setenv("AI_TRADING_DECISION_RECORD_SNAPSHOT_REDACT_SECRETS", "1")
    monkeypatch.setenv("AI_TRADING_CONFIG_SNAPSHOT_PATH", str(snapshot))

    bot_engine._write_decision_record(_DummyDecisionRecord(), str(primary))

    primary_payload = _read_single_json(primary)
    snapshot_payload = _read_single_json(snapshot)
    assert primary_payload["config_snapshot"]["api_key"] == "***"
    assert primary_payload["config_snapshot"]["nested"]["token"] == "***"
    assert snapshot_payload["config_snapshot"]["api_key"] == "***"
    assert snapshot_payload["config_snapshot"]["safe"] == "ok"


def test_write_decision_record_relative_path_uses_data_dir(
    tmp_path: Path, monkeypatch
) -> None:
    data_dir = tmp_path / "data"
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(data_dir))
    monkeypatch.setenv("AI_TRADING_DECISION_RECORD_SNAPSHOT_REDACT_SECRETS", "1")
    monkeypatch.setenv("AI_TRADING_CONFIG_SNAPSHOT_PATH", "")

    bot_engine._write_decision_record(
        _DummyDecisionRecord(),
        "runtime/decision_records.jsonl",
    )

    expected = (data_dir / "runtime" / "decision_records.jsonl").resolve()
    assert expected.exists()
    payload = _read_single_json(expected)
    assert payload["symbol"] == "AAPL"
    assert payload["config_snapshot"]["api_key"] == "***"


def test_write_decision_record_serializes_timestamp_like_values(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    primary = tmp_path / "decision_timestamp.jsonl"
    monkeypatch.setenv("AI_TRADING_DECISION_RECORD_SNAPSHOT_REDACT_SECRETS", "1")
    monkeypatch.setenv("AI_TRADING_CONFIG_SNAPSHOT_PATH", "")

    with caplog.at_level(logging.WARNING):
        bot_engine._write_decision_record(_TimestampDecisionRecord(), str(primary))

    payload = _read_single_json(primary)
    assert payload["symbol"] == "AAPL"
    assert isinstance(payload["bar_ts"], str)
    assert not any(record.getMessage() == "DECISION_RECORD_WRITE_FAILED" for record in caplog.records)


def test_tca_stale_block_reason_respects_latest_timestamp(
    tmp_path: Path, monkeypatch
) -> None:
    tca_path = tmp_path / "tca.jsonl"
    monkeypatch.setenv("AI_TRADING_BLOCK_TRADING_IF_TCA_STALE", "1")
    monkeypatch.setenv("AI_TRADING_TCA_PENDING_WRITE_SEC", "60")
    monkeypatch.setenv("AI_TRADING_TCA_PATH", str(tca_path))

    now = datetime.now(UTC)
    stale_record = {"ts": (now - timedelta(seconds=120)).isoformat(), "symbol": "AAPL"}
    tca_path.write_text(json.dumps(stale_record) + "\n", encoding="utf-8")
    assert bot_engine._tca_stale_block_reason(now) == "TCA_STALE_BLOCK"

    fresh_record = {"ts": now.isoformat(), "symbol": "AAPL"}
    with tca_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(fresh_record))
        handle.write("\n")
    assert bot_engine._tca_stale_block_reason(now) is None
