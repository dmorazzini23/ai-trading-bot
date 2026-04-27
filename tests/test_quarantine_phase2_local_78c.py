from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ai_trading.runtime import quarantine as q


def test_quarantine_from_dict_skips_invalid_entries_and_expiry() -> None:
    payload = {
        "sleeves": {
            "day": {
                "start_ts": "2026-04-27T10:00:00+00:00",
                "end_ts": "2026-04-27T11:00:00+00:00",
                "trigger_reason": "risk",
                "metrics_snapshot": {"reject_rate": 0.5},
            },
            "bad": {"start_ts": "not-a-date", "end_ts": "also-bad"},
            "ignored": "not-a-dict",
        },
        "symbols": {
            "aapl": {
                "start_ts": "2026-04-27T10:00:00",
                "end_ts": "2026-04-27T11:00:00",
                "trigger_reason": "loss",
                "metrics_snapshot": {"expectancy": -0.2},
            }
        },
    }

    manager = q.QuarantineManager.from_dict(payload)

    assert manager.is_quarantined(sleeve="day", now=datetime(2026, 4, 27, 10, 30, tzinfo=UTC)) == (
        True,
        "SLEEVE_QUARANTINED",
    )
    assert manager.is_quarantined(symbol="aapl", now=datetime(2026, 4, 27, 10, 30, tzinfo=UTC)) == (
        True,
        "SYMBOL_QUARANTINED",
    )
    assert manager.is_quarantined(sleeve="day", now=datetime(2026, 4, 27, 12, 0, tzinfo=UTC)) == (
        False,
        None,
    )
    assert "bad" not in manager.sleeves


def test_relative_path_resolution_uses_state_directory_and_repo_fallback(tmp_path, monkeypatch) -> None:
    state_dir = tmp_path / "state"
    monkeypatch.delenv("AI_TRADING_DATA_DIR", raising=False)
    monkeypatch.setenv("STATE_DIRECTORY", f"{state_dir}:ignored")
    assert q._resolve_state_path("runtime/quarantine.json") == (state_dir / "runtime/quarantine.json").resolve()

    monkeypatch.delenv("STATE_DIRECTORY", raising=False)
    resolved = q._resolve_state_path("runtime/quarantine.json")
    assert resolved.is_absolute()
    assert resolved.name == "quarantine.json"


def test_load_quarantine_state_invalid_primary_backup_and_restore_failure(tmp_path, monkeypatch) -> None:
    primary = tmp_path / "quarantine.json"
    backup = q._backup_state_path(primary)

    assert q.load_quarantine_state(str(primary)).to_dict() == {"sleeves": {}, "symbols": {}}

    primary.write_text("[]", encoding="utf-8")
    assert q.load_quarantine_state(str(primary)).to_dict() == {"sleeves": {}, "symbols": {}}

    manager = q.QuarantineManager()
    manager.quarantine_symbol(
        "MSFT",
        duration=timedelta(minutes=5),
        trigger_reason="backup",
        metrics_snapshot={},
    )
    primary.write_text("{", encoding="utf-8")
    backup.write_text(json.dumps(manager.to_dict()), encoding="utf-8")

    def fail_restore(_dest: Path, _payload: str) -> None:
        raise OSError("restore denied")

    monkeypatch.setattr(q, "_write_json_atomically", fail_restore)
    loaded = q.load_quarantine_state(str(primary))
    assert loaded.is_quarantined(symbol="MSFT")[0] is True


def test_save_quarantine_state_tolerates_backup_write_failure(tmp_path, monkeypatch) -> None:
    manager = q.QuarantineManager()
    manager.quarantine_sleeve(
        "overnight",
        duration=timedelta(minutes=1),
        trigger_reason="test",
        metrics_snapshot={"loss": 1},
    )
    calls: list[Path] = []
    original_write = q._write_json_atomically

    def flaky_write(dest: Path, payload: str) -> None:
        calls.append(dest)
        if dest.suffix == ".bak":
            raise OSError("backup failed")
        original_write(dest, payload)

    monkeypatch.setattr(q, "_write_json_atomically", flaky_write)
    path = tmp_path / "quarantine.json"
    q.save_quarantine_state(str(path), manager)

    assert path.exists()
    assert calls == [path, path.with_suffix(".json.bak")]
