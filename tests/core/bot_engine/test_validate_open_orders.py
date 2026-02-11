from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from ai_trading.core import bot_engine


def test_validate_open_orders_preserves_original_submit_error(monkeypatch):
    """Inner order-submit failures should not be masked by local logger scope issues."""

    monkeypatch.setattr(bot_engine, "_parse_local_positions", lambda: {"AAPL": 1})
    monkeypatch.setattr(bot_engine, "_ensure_alpaca_classes", lambda: None)
    monkeypatch.setattr(bot_engine, "audit_positions", lambda _ctx: None)
    monkeypatch.setattr(
        bot_engine,
        "list_open_orders",
        lambda _api: [
            SimpleNamespace(
                id="ord-1",
                symbol="AAPL",
                qty="1",
                side="buy",
                status="new",
                created_at=datetime.now(UTC) - timedelta(minutes=10),
            )
        ],
    )
    monkeypatch.setattr(
        bot_engine,
        "MarketOrderRequest",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        bot_engine,
        "safe_submit_order",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("submit failed")),
    )

    ctx = SimpleNamespace(api=SimpleNamespace(cancel_order=lambda _oid: None))

    with pytest.raises(ValueError, match="submit failed"):
        bot_engine.validate_open_orders(ctx)


def test_validate_open_orders_handles_fractional_qty_payload(monkeypatch):
    """Fractional quantity strings should not crash stale-order cleanup."""

    calls: dict[str, int] = {"submitted": 0}

    monkeypatch.setattr(bot_engine, "_parse_local_positions", lambda: {"AAPL": 1})
    monkeypatch.setattr(bot_engine, "_ensure_alpaca_classes", lambda: None)
    monkeypatch.setattr(bot_engine, "audit_positions", lambda _ctx: None)
    monkeypatch.setattr(
        bot_engine,
        "list_open_orders",
        lambda _api: [
            SimpleNamespace(
                id="ord-2",
                symbol="AAPL",
                qty="0.5",
                side="buy",
                status="new",
                created_at=datetime.now(UTC) - timedelta(minutes=10),
            )
        ],
    )
    monkeypatch.setattr(
        bot_engine,
        "safe_submit_order",
        lambda *_a, **_k: calls.__setitem__("submitted", calls["submitted"] + 1),
    )

    ctx = SimpleNamespace(api=SimpleNamespace(cancel_order=lambda _oid: None))
    bot_engine.validate_open_orders(ctx)

    assert calls["submitted"] == 0
