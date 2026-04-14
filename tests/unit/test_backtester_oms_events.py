from __future__ import annotations

from pathlib import Path

import pytest

from ai_trading.oms.event_store import EventStore
from ai_trading.strategies.backtester import BacktestEngine, DefaultExecutionModel


pd = pytest.importorskip("pandas")
pytest.importorskip("sqlalchemy")


def test_backtester_emits_oms_lifecycle_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "backtest_oms_events.db"
    monkeypatch.setenv("AI_TRADING_BACKTEST_OMS_EVENTS_ENABLED", "1")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("AI_TRADING_OMS_EVENT_JSONL_ENABLED", "0")

    index = pd.date_range("2025-01-01", periods=12, freq="D")
    closes = [100, 100, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    frame = pd.DataFrame(
        {
            "open": closes,
            "high": [value + 0.5 for value in closes],
            "low": [value - 0.5 for value in closes],
            "close": closes,
            "volume": [2000] * len(closes),
        },
        index=index,
    )

    engine = BacktestEngine({"AAPL": frame}, DefaultExecutionModel())
    result = engine.run(["AAPL"])
    assert not result.trades.empty

    event_store = EventStore(url=f"sqlite:///{db_path}")
    rows = event_store.list_oms_events(limit=5000)
    event_store.close()
    assert rows
    sources = {str(row.get("event_source")) for row in rows}
    assert "intent_store" in sources
    event_types = {str(row.get("event_type")) for row in rows}
    assert "INTENT_CREATED" in event_types
    assert "SUBMIT_CLAIMED" in event_types
    assert "SUBMIT_ATTEMPTED" in event_types
    assert "SUBMIT_ACK" in event_types
    assert "ORDER_PARTIALLY_FILLED" in event_types
    assert "ORDER_FILLED" in event_types
    assert "INTENT_CLOSED" in event_types

    created_intents = {
        str(row.get("intent_id"))
        for row in rows
        if str(row.get("event_type")) == "INTENT_CREATED"
    }
    closed_intents = {
        str(row.get("intent_id"))
        for row in rows
        if str(row.get("event_type")) == "INTENT_CLOSED"
    }
    assert created_intents
    assert created_intents.issubset(closed_intents)
