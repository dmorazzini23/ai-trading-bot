from __future__ import annotations

from datetime import UTC, datetime

from ai_trading.oms.pretrade import SlidingWindowRateLimiter


def test_durable_rate_limiter_shares_global_budget_across_instances(tmp_path) -> None:
    state_path = tmp_path / "pretrade_rate_limiter.db"
    first = SlidingWindowRateLimiter(global_orders_per_min=1, state_path=state_path)
    second = SlidingWindowRateLimiter(global_orders_per_min=1, state_path=state_path)
    bar_ts = datetime(2026, 4, 21, 14, 30, tzinfo=UTC)

    first_allowed, first_reason, first_details = first.allow_and_record_order("AAPL", bar_ts)
    second_allowed, second_reason, second_details = second.allow_and_record_order("MSFT", bar_ts)

    assert first_allowed is True
    assert first_reason is None
    assert first_details == {}
    assert second_allowed is False
    assert second_reason == "RATE_THROTTLE_BLOCK"
    assert second_details["scope"] == "global"


def test_durable_rate_limiter_shares_cancel_loop_state_across_instances(tmp_path) -> None:
    state_path = tmp_path / "pretrade_rate_limiter.db"
    first = SlidingWindowRateLimiter(
        cancel_loop_max_without_fill=1,
        cancel_loop_block_bars=2,
        state_path=state_path,
    )
    second = SlidingWindowRateLimiter(
        cancel_loop_max_without_fill=1,
        cancel_loop_block_bars=2,
        state_path=state_path,
    )
    first_bar = datetime(2026, 4, 21, 14, 30, tzinfo=UTC)
    second_bar = datetime(2026, 4, 21, 14, 31, tzinfo=UTC)

    first.record_cancel("AAPL", bar_ts=first_bar, filled=False)
    allowed, reason, details = second.allow_order("AAPL", second_bar)

    assert allowed is False
    assert reason == "CANCEL_LOOP_BLOCK"
    assert details["symbol"] == "AAPL"
