from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from ai_trading.oms.pretrade import OrderIntent, SlidingWindowRateLimiter, validate_pretrade


class _Ledger:
    @staticmethod
    def seen_client_order_id(_value: str) -> bool:
        return True


def test_pretrade_blocks_duplicate_order_id() -> None:
    cfg = SimpleNamespace(
        max_order_dollars=100000.0,
        max_order_shares=10000,
        price_collar_pct=0.1,
    )
    intent = OrderIntent(
        symbol="NVDA",
        side="sell",
        qty=5,
        notional=500.0,
        limit_price=100.0,
        bar_ts=datetime.now(UTC),
        client_order_id="dup-cid",
        last_price=100.0,
        mid=100.0,
    )
    limiter = SlidingWindowRateLimiter(global_orders_per_min=100, per_symbol_orders_per_min=100)
    allowed, reason, details = validate_pretrade(intent, cfg=cfg, ledger=_Ledger(), rate_limiter=limiter)
    assert allowed is False
    assert reason == "DUPLICATE_ORDER_BLOCK"
    assert details["client_order_id"] == "dup-cid"
