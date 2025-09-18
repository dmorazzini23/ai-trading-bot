from __future__ import annotations

from types import SimpleNamespace

import ai_trading.meta_learning as meta_learning
from ai_trading.core import bot_engine


def test_trade_logger_triggers_conversion_on_audit_rows(monkeypatch, tmp_path):
    reward_log = tmp_path / "reward_log.csv"
    trade_log = tmp_path / "trades.csv"

    monkeypatch.setattr(bot_engine, "REWARD_LOG_FILE", str(reward_log))

    trade_logger = bot_engine.TradeLogger(path=trade_log)
    state = bot_engine.BotState()
    state.capital_band = "small"

    trade_logger.log_entry(
        "AAPL",
        price=100.0,
        qty=1,
        side="buy",
        strategy="test_strategy",
        confidence=0.6,
    )

    monkeypatch.setattr(
        bot_engine,
        "get_settings",
        lambda: SimpleNamespace(enable_sklearn=True),
    )
    monkeypatch.setattr(
        "ai_trading.config.get_settings",
        lambda: SimpleNamespace(enable_sklearn=True),
    )

    conversion_calls: list[dict[str, object]] = []

    def fake_trigger(trade_data: dict[str, object]) -> bool:
        conversion_calls.append(trade_data)
        return True

    def fake_quality_report(_path: str) -> dict[str, int]:
        return {"audit_format_rows": 2, "meta_format_rows": 0}

    monkeypatch.setattr(meta_learning, "trigger_meta_learning_conversion", fake_trigger)
    monkeypatch.setattr(meta_learning, "validate_trade_data_quality", fake_quality_report)

    trade_logger.log_exit(state, "AAPL", 105.0)

    assert len(conversion_calls) == 1
    trade_data = conversion_calls[0]
    assert trade_data.get("symbol") == "AAPL"
    assert trade_data.get("exit_price") == 105.0
