from __future__ import annotations

import json
from pathlib import Path

from ai_trading.tools import seed_trade_history


def test_default_seed_trade_history_filename_is_explicit() -> None:
    assert seed_trade_history.DEFAULT_PATH == "trade_history.seed.json"


def test_load_history_uses_default_seed_file(tmp_path, monkeypatch) -> None:
    records = [
        {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 0,
            "entry_price": 0,
            "exit_price": 0,
            "entry_time": "1970-01-01T00:00:00",
            "exit_time": "1970-01-01T00:00:00",
            "pnl": 0,
            "confidence": 0.0,
            "signal_strength": 0.0,
        }
    ]
    (tmp_path / seed_trade_history.DEFAULT_PATH).write_text(json.dumps(records))
    monkeypatch.chdir(tmp_path)

    assert seed_trade_history.load_history() == records


def test_load_history_uses_packaged_default_when_local_file_missing(monkeypatch) -> None:
    monkeypatch.chdir(Path("/tmp"))

    records = seed_trade_history.load_history()

    assert isinstance(records, list)
    assert records
