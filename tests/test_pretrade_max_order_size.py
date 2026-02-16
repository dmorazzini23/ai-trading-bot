from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from ai_trading.oms.pretrade import OrderIntent, SlidingWindowRateLimiter, validate_pretrade


class _Ledger:
    @staticmethod
    def seen_client_order_id(_value: str) -> bool:
        return False


def test_pretrade_blocks_order_size() -> None:
    cfg = SimpleNamespace(
        max_order_dollars=1000.0,
        max_order_shares=5,
        price_collar_pct=0.1,
    )
    intent = OrderIntent(
        symbol="MSFT",
        side="buy",
        qty=10,
        notional=2500.0,
        limit_price=250.0,
        bar_ts=datetime.now(UTC),
        client_order_id="cid-2",
        last_price=250.0,
        mid=250.0,
    )
    limiter = SlidingWindowRateLimiter(global_orders_per_min=100, per_symbol_orders_per_min=100)
    allowed, reason, details = validate_pretrade(intent, cfg=cfg, ledger=_Ledger(), rate_limiter=limiter)
    assert allowed is False
    assert reason == "ORDER_SIZE_BLOCK"
    assert details["qty"] == 10
