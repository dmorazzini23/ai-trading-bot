"""The outer bot-engine wrapper must not repeat an uncertain broker submit."""

from types import SimpleNamespace

import pytest

from ai_trading.core import bot_engine


def test_outer_submit_does_not_retry_broker_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[str] = []

    def uncertain_submit(_ctx, symbol, _qty, _side, **_kwargs):
        attempts.append(symbol)
        raise bot_engine.APIError("broker response lost")

    monkeypatch.setattr(bot_engine, "_submit_order_service", uncertain_submit)

    with pytest.raises(bot_engine.APIError):
        bot_engine.submit_order(SimpleNamespace(), "AAPL", 1, "buy", price=100.0)

    assert attempts == ["AAPL"]
