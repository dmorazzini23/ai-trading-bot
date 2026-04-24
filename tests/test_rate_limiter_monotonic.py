from __future__ import annotations

import threading

from ai_trading.integrations.rate_limit import RateLimiter
from ai_trading.data.rate_limiter import TokenBucket as DataTokenBucket
from ai_trading.integrations.rate_limit import (
    RateLimitConfig,
    TokenBucket as IntegrationTokenBucket,
)


def test_data_token_bucket_refill_uses_monotonic_clock(monkeypatch) -> None:
    wall_values = iter([1_000.0, 10.0])
    mono_values = iter([100.0, 103.0])
    monkeypatch.setattr("ai_trading.data.rate_limiter.time.time", lambda: next(wall_values))
    monkeypatch.setattr(
        "ai_trading.data.rate_limiter.time.monotonic",
        lambda: next(mono_values),
    )

    bucket = DataTokenBucket(rate=2.0, capacity=10, name="test")
    bucket.tokens = 0.0
    bucket._refill()

    assert bucket.tokens == 6.0


def test_integration_token_bucket_refill_uses_monotonic_clock(monkeypatch) -> None:
    wall_values = iter([1_000.0, 10.0])
    mono_values = iter([100.0, 104.0])
    monkeypatch.setattr("ai_trading.integrations.rate_limit.time.time", lambda: next(wall_values))
    monkeypatch.setattr(
        "ai_trading.integrations.rate_limit.time.monotonic",
        lambda: next(mono_values),
    )
    monkeypatch.setattr(
        "ai_trading.integrations.rate_limit.random.random",
        lambda: 0.5,
    )

    bucket = IntegrationTokenBucket(capacity=10, refill_rate=2.0)
    bucket.tokens = 0.0
    bucket._refill()

    assert bucket.tokens == 8.0


def test_rate_limiter_all_route_status_does_not_reenter_lock() -> None:
    limiter = RateLimiter(global_capacity=10, global_rate=10.0)
    limiter.configure_route("custom", RateLimitConfig(capacity=1, refill_rate=1.0))
    result: dict[str, object] = {}

    def read_status() -> None:
        result["status"] = limiter.get_status()

    thread = threading.Thread(target=read_status)
    thread.start()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    status = result["status"]
    assert isinstance(status, dict)
    assert "custom" in status["routes"]
