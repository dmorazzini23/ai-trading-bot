from __future__ import annotations

import csv
import logging


def test_trade_log_cached_once(tmp_path, monkeypatch):
    import ai_trading.core.bot_engine as bot_engine

    trade_log_path = tmp_path / "trades.csv"
    rows = [
        ["exit_price", "entry_price", "signal_tags", "side"],
        ["150", "140", "momentum", "buy"],
        ["152", "138", "momentum", "buy"],
    ]
    with trade_log_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)

    monkeypatch.setattr(bot_engine, "TRADE_LOG_FILE", str(trade_log_path), raising=False)
    bot_engine._TRADE_LOGGER_SINGLETON = None
    bot_engine._TRADE_LOG_CACHE = None
    bot_engine._TRADE_LOG_CACHE_LOADED = False
    bot_engine._EMPTY_TRADE_LOG_INFO_EMITTED = False
    bot_engine.signal_manager._cycle_trade_log = None
    bot_engine.signal_manager._cycle_trade_log_cycle_id = None
    bot_engine.signal_manager._cycle_trade_log_source = None

    bot_engine._reset_cycle_cache()

    logger = logging.getLogger("ai_trading.core.bot_engine")

    class _ListHandler(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.events: list[str] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.events.append(record.getMessage())

    handler = _ListHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        bot_engine.signal_manager.begin_cycle()
        bot_engine.load_global_signal_performance(min_trades=1)
        bot_engine.load_global_signal_performance(min_trades=1)
    finally:
        logger.removeHandler(handler)

    messages = handler.events
    assert sum(msg.startswith("TRADE_LOG_CACHED") for msg in messages) == 1
    assert sum(msg.startswith("TRADE_LOG_LOADED") for msg in messages) == 1
