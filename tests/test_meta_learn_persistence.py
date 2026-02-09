from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from ai_trading.execution.engine import (
    ExecutionEngine,
    Order,
    OrderSide,
    OrderType,
    _SignalMeta,
)
from ai_trading.math.money import Money

pd = pytest.importorskip("pandas")


def test_trade_persistence_updates_canonical_history(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    canonical_path = tmp_path / "trade_history.parquet"
    trade_log_path = tmp_path / "trades.csv"

    import ai_trading.meta_learning.persistence as persistence

    monkeypatch.setattr(persistence, "_CANONICAL_PATH", canonical_path)
    monkeypatch.setattr(persistence, "_PANDAS_MISSING_LOGGED", False)

    import ai_trading.core.bot_engine as bot_engine

    monkeypatch.setattr(bot_engine, "TRADE_LOG_FILE", str(trade_log_path), raising=False)
    bot_engine._TRADE_LOGGER_SINGLETON = None
    bot_engine._TRADE_LOG_CACHE = None
    bot_engine._TRADE_LOG_CACHE_LOADED = False
    bot_engine._EMPTY_TRADE_LOG_INFO_EMITTED = False

    caplog.set_level(logging.INFO)
    first_read = bot_engine._read_trade_log(str(trade_log_path))
    assert first_read is None
    empty_logs = [rec for rec in caplog.records if "TRADE_LOG_EMPTY" in rec.getMessage()]
    assert empty_logs, "expected empty trade log notification"

    ctx = SimpleNamespace(risk_engine=SimpleNamespace(register_fill=lambda *_: None))
    engine = ExecutionEngine(ctx=ctx)
    order = Order("AAPL", OrderSide.BUY, 100, order_type=OrderType.MARKET, price=Money(150))
    order.add_fill(100, Money(151))
    signal = SimpleNamespace(
        symbol="AAPL",
        side="buy",
        strategy="alpha",
        confidence=0.8,
        signal_tags="alpha",
        weight=0.5,
    )
    engine._order_signal_meta[order.id] = _SignalMeta(signal=signal, requested_qty=100, signal_weight=0.5)
    caplog.clear()
    engine._handle_execution_event(order, "completed")

    assert canonical_path.exists()
    frame = pd.read_parquet(canonical_path)
    assert len(frame) >= 1

    caplog.clear()
    loaded = bot_engine._read_trade_log(str(trade_log_path))
    assert loaded is not None
    assert len(loaded) >= 1
    loaded_logs = [rec for rec in caplog.records if rec.getMessage().startswith("TRADE_LOG_LOADED")]
    assert loaded_logs, "expected TRADE_LOG_LOADED log"
    fallback_logs = [rec for rec in caplog.records if "TRADE_LOG_EMPTY" in rec.getMessage()]
    assert not fallback_logs, "empty log should not repeat after trades"


def test_load_trade_history_falls_back_to_pickle_when_parquet_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    canonical_path = tmp_path / "trade_history.parquet"

    import ai_trading.meta_learning.persistence as persistence

    monkeypatch.setattr(persistence, "_CANONICAL_PATH", canonical_path)
    monkeypatch.setattr(persistence, "_READ_FAILURE_LOGGED", set(), raising=False)
    monkeypatch.setattr(persistence, "_pytest_active", lambda: False)

    expected = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "entry_time": pd.Timestamp("2024-01-01T14:30:00+00:00"),
                "entry_price": 100.0,
                "qty": 1,
                "side": "buy",
            }
        ]
    )
    expected.to_pickle(canonical_path)

    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'.")
        ),
    )

    frame, source = persistence.load_trade_history(sync_from_broker=False)

    assert source == "canonical"
    assert frame is not None
    assert not frame.empty
    assert frame.iloc[0]["symbol"] == "AAPL"


def test_trade_history_engine_missing_log_emits_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    canonical_path = tmp_path / "trade_history.parquet"

    import ai_trading.meta_learning.persistence as persistence

    monkeypatch.setattr(persistence, "_CANONICAL_PATH", canonical_path)
    monkeypatch.setattr(persistence, "_READ_FAILURE_LOGGED", set(), raising=False)
    monkeypatch.setattr(persistence, "_pytest_active", lambda: False)

    canonical_path.write_text("not-a-valid-trade-history", encoding="utf-8")
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'.")
        ),
    )

    caplog.set_level(logging.INFO, logger=persistence.logger.name)

    frame_first, source_first = persistence.load_trade_history(sync_from_broker=False)
    frame_second, source_second = persistence.load_trade_history(sync_from_broker=False)

    assert frame_first is None and source_first is None
    assert frame_second is None and source_second is None
    engine_logs = [
        rec for rec in caplog.records if rec.message == "TRADE_HISTORY_PARQUET_ENGINE_MISSING"
    ]
    assert len(engine_logs) == 1
