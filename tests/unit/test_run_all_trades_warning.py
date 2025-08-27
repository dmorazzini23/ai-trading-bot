"""Ensure run_all_trades_worker does not warn for a valid API client."""

from __future__ import annotations

import types
from unittest.mock import Mock, call

import ai_trading.core.bot_engine as eng


def test_run_all_trades_no_warning_with_valid_api(monkeypatch):
    """A valid client with get_orders should not trigger a warning."""

    class DummyAPI:
        def __init__(self):
            self.called_with: dict | None = None

        def get_orders(self, **kwargs):
            self.called_with = kwargs
            return []

    class DummyRiskEngine:
        def wait_for_exposure_update(self, timeout: float) -> None:  # noqa: D401
            """No-op risk update."""

    state = eng.BotState()
    api = DummyAPI()
    runtime = types.SimpleNamespace(api=api, risk_engine=DummyRiskEngine())

    # Minimal patches to isolate the order-check logic
    monkeypatch.setattr(eng, "_ensure_alpaca_classes", lambda: None)
    monkeypatch.setattr(eng, "_init_metrics", lambda: None)
    monkeypatch.setattr(eng, "is_market_open", lambda: True)
    monkeypatch.setattr(eng, "ensure_alpaca_attached", lambda runtime: None)
    monkeypatch.setattr(eng, "check_pdt_rule", lambda runtime: False)
    monkeypatch.setattr(eng, "get_strategies", lambda: [])
    monkeypatch.setattr(eng, "get_verbose_logging", lambda: False)

    def _raise_df(*_a, **_k):
        raise eng.DataFetchError("boom")

    monkeypatch.setattr(eng, "_prepare_run", _raise_df)
    monkeypatch.setattr(eng.CFG, "log_market_fetch", False, raising=False)

    class DummyLock:
        def acquire(self, blocking: bool = False) -> bool:  # noqa: D401
            """Always acquire."""

            return True

        def release(self) -> None:
            """No-op release."""

    monkeypatch.setattr(eng, "run_lock", DummyLock())

    warn_mock = Mock()
    monkeypatch.setattr(eng.logger_once, "warning", warn_mock)

    eng.run_all_trades_worker(state, runtime)

    warn_mock.assert_has_calls(
        [
            call("API_GET_ORDERS_MAPPED", key="alpaca_get_orders_mapped"),
            call("ALPACA_API_ADAPTER", key="alpaca_api_adapter"),
        ]
    )
    assert api.called_with is not None
    assert "statuses" in api.called_with

