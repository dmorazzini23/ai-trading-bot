from __future__ import annotations

import logging

from ai_trading.config import runtime


def test_trading_config_logs_feed_ignored_for_non_alpaca_provider(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=runtime.logger.name)
    monkeypatch.setattr(runtime, "_FEED_IGNORE_LOGGED", False, raising=False)

    cfg = runtime.TradingConfig.from_env(
        {
            "DATA_PROVIDER": "finnhub",
            "ALPACA_DATA_FEED": "sip",
            "MAX_DRAWDOWN_THRESHOLD": "0.08",
        }
    )

    matching = [
        record
        for record in caplog.records
        if record.getMessage() == "FEED_IGNORED_NON_ALPACA_PROVIDER"
    ]
    assert matching, "expected FEED_IGNORED_NON_ALPACA_PROVIDER log"
    assert getattr(cfg, "alpaca_feed_ignored", False) is True
    assert cfg.alpaca_data_feed == "sip"
