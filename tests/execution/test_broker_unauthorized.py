from __future__ import annotations

from types import SimpleNamespace

import pytest

from ai_trading.execution import live_trading as lt


@pytest.fixture
def execution_engine(monkeypatch):
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.setenv("EXECUTION_MODE", "paper")

    monkeypatch.setattr(lt, "get_alpaca_creds", lambda: ("key", "secret"))
    monkeypatch.setattr(lt, "get_alpaca_base_url", lambda: "https://paper-api.alpaca.markets")

    exec_settings = SimpleNamespace(
        mode="paper",
        shadow_mode=False,
        order_timeout_seconds=30,
        slippage_limit_bps=10,
        price_provider_order=("alpaca",),
        data_feed_intraday="iex",
    )
    monkeypatch.setattr(lt, "get_execution_settings", lambda: exec_settings)

    trading_config = SimpleNamespace(
        safe_mode_allow_paper=False,
        execution_mode="paper",
        shadow_mode=False,
    )
    monkeypatch.setattr(lt, "get_trading_config", lambda: trading_config)
    monkeypatch.setattr(lt, "get_settings", lambda: SimpleNamespace(halt_flag_path="halt.flag"))

    engine = lt.ExecutionEngine()
    engine.is_initialized = True
    engine._ensure_initialized = lambda: True
    engine._pre_execution_checks = lambda: True
    engine.trading_client = SimpleNamespace()
    engine._get_account_snapshot = lambda: {}
    engine._should_skip_for_pdt = lambda account, closing: (False, None, {})
    return engine


def test_broker_unauthorized_engages_backoff(monkeypatch, execution_engine):
    engine = execution_engine

    monkeypatch.setattr(
        lt,
        "_call_preflight_capacity",
        lambda _symbol, _side, _price, qty, *_args: lt.CapacityCheck(True, qty, None),
    )

    class UnauthorizedError(lt.APIError):
        def __init__(self, message: str = "401 unauthorized") -> None:
            super().__init__(message)

        @property
        def status_code(self) -> int:  # type: ignore[override]
            return 401

        @property
        def code(self) -> str:  # type: ignore[override]
            return "40110000"

        @property
        def message(self) -> str:  # type: ignore[override]
            return "401 unauthorized"

    def raise_unauthorized(_order):
        raise UnauthorizedError()

    engine._submit_order_to_alpaca = raise_unauthorized  # type: ignore[assignment]

    result = engine.submit_market_order("AAPL", "buy", 1)

    assert result is None
    assert engine._broker_lock_reason == "unauthorized"
    lock_remaining = engine._broker_locked_until - lt.monotonic_time()
    assert lock_remaining >= 119.0

    called: list[bool] = []

    def should_not_execute(order):
        called.append(True)
        return {"id": "should_not"}

    engine._submit_order_to_alpaca = should_not_execute  # type: ignore[assignment]

    result_second = engine.submit_market_order("AAPL", "buy", 1)

    assert result_second is None
    assert not called
    assert engine.stats["skipped_orders"] >= 1
