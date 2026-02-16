from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from ai_trading.oms.pretrade import OrderIntent, SlidingWindowRateLimiter, validate_pretrade


class _Ledger:
    @staticmethod
    def seen_client_order_id(_value: str) -> bool:
        return False


def test_pretrade_blocks_price_collar() -> None:
    cfg = SimpleNamespace(
        max_order_dollars=100000.0,
        max_order_shares=10000,
        price_collar_pct=0.05,
    )
    intent = OrderIntent(
        symbol="AAPL",
        side="buy",
        qty=10,
        notional=1200.0,
        limit_price=120.0,
        bar_ts=datetime.now(UTC),
        client_order_id="cid-1",
        last_price=100.0,
        mid=100.0,
    )
    limiter = SlidingWindowRateLimiter(global_orders_per_min=100, per_symbol_orders_per_min=50)
    allowed, reason, _details = validate_pretrade(intent, cfg=cfg, ledger=_Ledger(), rate_limiter=limiter)
    assert allowed is False
    assert reason == "PRICE_COLLAR_BLOCK"
