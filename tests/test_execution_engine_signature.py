"""Regression tests for the live execution engine public signature."""

from __future__ import annotations

from typing import Any

import pytest

from ai_trading.execution.live_trading import ExecutionEngine


@pytest.fixture()
def engine(monkeypatch: pytest.MonkeyPatch) -> ExecutionEngine:
    """Create an execution engine with broker calls patched out."""

    engine = ExecutionEngine()

    def fake_market(symbol: str, side: str, quantity: int, **kwargs: Any) -> dict[str, Any]:
        fake_market.captured = kwargs  # type: ignore[attr-defined]
        return {
            "id": "mock-market-1",
            "status": "accepted",
            "qty": quantity,
        }

    def fake_limit(
        symbol: str,
        side: str,
        quantity: int,
        limit_price: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        fake_limit.captured = kwargs  # type: ignore[attr-defined]
        return {
            "id": "mock-limit-1",
            "status": "accepted",
            "qty": quantity,
            "limit_price": limit_price,
        }

    monkeypatch.setattr(engine, "submit_market_order", fake_market)
    monkeypatch.setattr(engine, "submit_limit_order", fake_limit)
    return engine


def test_execute_order_forwards_supported_asset_class(engine: ExecutionEngine, monkeypatch: pytest.MonkeyPatch) -> None:
    """When supported, ``asset_class`` is forwarded to the broker request."""

    monkeypatch.setattr(engine, "_supports_asset_class", lambda: True)
    result = engine.execute_order(
        "AAPL",
        "buy",
        5,
        order_type="limit",
        limit_price=123.45,
        asset_class="us_equity",
    )
    assert str(result) == "mock-limit-1"
    captured = getattr(engine.submit_limit_order, "captured")  # type: ignore[attr-defined]
    assert captured.get("asset_class") == "us_equity"


def test_execute_order_ignores_unknown_kwargs(
    engine: ExecutionEngine, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Unknown kwargs are ignored gracefully with a debug breadcrumb."""

    caplog.set_level("DEBUG", logger="ai_trading.execution.live_trading")
    monkeypatch.setattr(engine, "_supports_asset_class", lambda: False)

    result = engine.execute_order(
        "AAPL",
        "buy",
        10,
        order_type="market",
        asset_class="us_equity",
        foo="bar",
    )
    assert str(result) == "mock-market-1"
    captured = getattr(engine.submit_market_order, "captured")  # type: ignore[attr-defined]
    assert "asset_class" not in captured
    assert "foo" not in captured

    ignored = [record for record in caplog.records if record.msg == "EXEC_IGNORED_KWARG"]
    logged_fields = {getattr(record, "kw", None) for record in ignored}
    assert logged_fields >= {"asset_class", "foo"}
