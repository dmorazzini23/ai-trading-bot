from __future__ import annotations

import builtins
import csv
from pathlib import Path

from ai_trading import audit


def test_log_trade_writes_simple_audit_schema_with_exposure(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "audit.csv"
    monkeypatch.setenv("TRADE_LOG_FILE", str(path))
    monkeypatch.setattr(audit, "TRADE_LOG_FILE", str(path))

    audit.log_trade("SPY", 3, "buy", 101.25, timestamp="2026-01-01T14:30:00Z", extra_info="AUDIT", exposure=42.5)

    rows = list(csv.DictReader(path.open(newline="")))
    assert rows == [
        {
            "id": rows[0]["id"],
            "timestamp": "2026-01-01T14:30:00Z",
            "symbol": "SPY",
            "side": "buy",
            "qty": "3",
            "price": "101.25",
            "exposure": "42.5",
            "mode": "AUDIT",
            "result": "",
        }
    ]
    assert rows[0]["id"]


def test_compute_targets_honors_explicit_env_and_deduplicates(monkeypatch, tmp_path: Path) -> None:
    explicit = tmp_path / "explicit.csv"
    monkeypatch.setenv("TRADE_LOG_FILE", str(explicit))
    monkeypatch.setenv("PYTEST_RUNNING", "1")

    assert audit._compute_targets(explicit) == [explicit]

    monkeypatch.delenv("TRADE_LOG_FILE", raising=False)
    targets = audit._compute_targets(tmp_path / "trades.csv")

    assert targets[0] == tmp_path / "trades.csv"
    assert len(targets) == len(set(targets))


def test_log_trade_retries_write_after_permission_repair(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "trades.csv"
    monkeypatch.setenv("TRADE_LOG_FILE", str(path))
    monkeypatch.setattr(audit, "TRADE_LOG_FILE", str(path))
    monkeypatch.setattr(audit, "_ensure_parent_dir", lambda _path: None)

    real_open = builtins.open
    monkeypatch.setattr(
        audit,
        "_ensure_file_header",
        lambda _path, headers: path.write_text(",".join(headers) + "\n"),
    )
    attempts = {"open": 0, "fix": 0}

    def flaky_open(*args, **kwargs):
        if args and args[0] == path and args[1] == "a" and attempts["open"] == 0:
            attempts["open"] += 1
            raise PermissionError("denied")
        attempts["open"] += 1
        return real_open(*args, **kwargs)

    def repair(_path, *_headers):
        attempts["fix"] += 1
        return True

    monkeypatch.setattr(builtins, "open", flaky_open)
    monkeypatch.setattr(audit, "fix_file_permissions", repair)

    audit.log_trade("MSFT", 2, "sell", 99.5, timestamp="t", extra_info="strategy")

    assert attempts["fix"] == 1
    assert attempts["open"] >= 2
    assert "MSFT" in path.read_text()
