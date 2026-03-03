from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from ai_trading.oms.pretrade import (
    OrderIntent,
    SlidingWindowRateLimiter,
    safe_validate_pretrade,
    validate_pretrade,
)


class _ExposureLedger:
    def __init__(self, *, symbol_qty: dict[str, float] | None = None, gross_notional: float = 0.0) -> None:
        self._symbol_qty = {str(k).upper(): float(v) for k, v in (symbol_qty or {}).items()}
        self._gross_notional = float(gross_notional)

    @staticmethod
    def seen_client_order_id(_value: str) -> bool:
        return False

    def position_qty(self, symbol: str) -> float:
        return float(self._symbol_qty.get(str(symbol).upper(), 0.0))

    def gross_notional(self) -> float:
        return float(self._gross_notional)


def _intent(*, symbol: str = "AAPL", side: str = "buy", qty: int = 1, price: float = 100.0) -> OrderIntent:
    return OrderIntent(
        symbol=symbol,
        side=side,
        qty=qty,
        notional=abs(float(qty) * float(price)),
        limit_price=price,
        bar_ts=datetime.now(UTC),
        client_order_id=f"{symbol.lower()}-{side}-{qty}",
        last_price=price,
        mid=price,
    )


def test_pretrade_blocks_projected_symbol_notional() -> None:
    cfg = SimpleNamespace(
        max_order_dollars=0.0,
        max_order_shares=0,
        price_collar_pct=0.10,
        max_symbol_notional=1000.0,
        max_gross_notional=0.0,
    )
    ledger = _ExposureLedger(symbol_qty={"AAPL": 8}, gross_notional=1200.0)
    intent = _intent(symbol="AAPL", side="buy", qty=5, price=100.0)
    limiter = SlidingWindowRateLimiter(global_orders_per_min=100, per_symbol_orders_per_min=100)

    allowed, reason, details = validate_pretrade(intent, cfg=cfg, ledger=ledger, rate_limiter=limiter)

    assert allowed is False
    assert reason == "SYMBOL_NOTIONAL_BLOCK"
    assert details["symbol"] == "AAPL"
    assert details["projected_symbol_notional"] == pytest.approx(1300.0)


def test_pretrade_blocks_projected_gross_notional() -> None:
    cfg = SimpleNamespace(
        max_order_dollars=0.0,
        max_order_shares=0,
        price_collar_pct=0.10,
        max_symbol_notional=0.0,
        max_gross_notional=5000.0,
    )
    ledger = _ExposureLedger(symbol_qty={"AAPL": 2}, gross_notional=4900.0)
    intent = _intent(symbol="AAPL", side="buy", qty=1, price=200.0)
    limiter = SlidingWindowRateLimiter(global_orders_per_min=100, per_symbol_orders_per_min=100)

    allowed, reason, details = validate_pretrade(intent, cfg=cfg, ledger=ledger, rate_limiter=limiter)

    assert allowed is False
    assert reason == "GROSS_NOTIONAL_BLOCK"
    assert details["projected_gross_notional"] == pytest.approx(5100.0)
    assert details["max_gross_notional"] == pytest.approx(5000.0)


def test_safe_validate_pretrade_fail_closed_blocks_on_gateway_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_TRADING_PRETRADE_FAIL_CLOSED", "1")
    intent = _intent(symbol="MSFT", side="buy", qty=1, price=300.0)
    cfg = SimpleNamespace(max_order_dollars=0.0, max_order_shares=0, price_collar_pct=0.10)

    allowed, reason, details = safe_validate_pretrade(
        intent,
        cfg=cfg,
        ledger=_ExposureLedger(),
        rate_limiter=None,  # type: ignore[arg-type]
    )

    assert allowed is False
    assert reason == "PRETRADE_VALIDATION_ERROR"
    assert details["fail_closed"] is True


def test_safe_validate_pretrade_fail_open_allows_on_gateway_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_TRADING_PRETRADE_FAIL_CLOSED", "0")
    intent = _intent(symbol="MSFT", side="buy", qty=1, price=300.0)
    cfg = SimpleNamespace(max_order_dollars=0.0, max_order_shares=0, price_collar_pct=0.10)

    allowed, reason, details = safe_validate_pretrade(
        intent,
        cfg=cfg,
        ledger=_ExposureLedger(),
        rate_limiter=None,  # type: ignore[arg-type]
    )

    assert allowed is True
    assert reason == "PRETRADE_VALIDATION_FAIL_OPEN"
    assert details["fail_closed"] is False
