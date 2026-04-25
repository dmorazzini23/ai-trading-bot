from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Callable

from ai_trading.data import rate_limiter


def _monotonic_values(values: list[float]) -> Callable[[], float]:
    remaining = iter(values)
    last = values[-1]

    def _next() -> float:
        nonlocal last
        try:
            last = next(remaining)
        except StopIteration:
            return last
        return last

    return _next


def test_token_bucket_consumes_refills_and_reports_status(monkeypatch) -> None:
    monkeypatch.setattr(rate_limiter.time, "monotonic", _monotonic_values([0.0, 0.0, 0.5, 0.5, 0.5]))

    bucket = rate_limiter.TokenBucket(rate=2.0, capacity=3, name="bars")

    assert bucket.consume(tokens=2, block=False) is True
    assert bucket.tokens == 1.0
    assert bucket.consume(tokens=3, block=False) is False

    status = bucket.get_status()

    assert status == {
        "name": "bars",
        "tokens_available": 1.0,
        "capacity": 3,
        "rate_per_second": 2.0,
        "utilization_pct": 66.7,
    }


def test_token_bucket_timeout_and_blocking_wait(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(
        rate_limiter.time,
        "monotonic",
        _monotonic_values([0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    monkeypatch.setattr(rate_limiter.time, "sleep", lambda seconds: sleeps.append(seconds))
    timeout_bucket = rate_limiter.TokenBucket(rate=1.0, capacity=1, name="orders")

    assert timeout_bucket.consume(tokens=1, block=False) is True
    assert timeout_bucket.consume(tokens=1, block=True, timeout=0.5) is False

    monkeypatch.setattr(
        rate_limiter.time,
        "monotonic",
        _monotonic_values([0.0, 0.0, 0.0, 2.0]),
    )
    blocking_bucket = rate_limiter.TokenBucket(rate=1.0, capacity=1, name="orders")
    blocking_bucket.tokens = 0.0
    blocking_bucket.last_update = 0.0

    assert blocking_bucket.consume(tokens=1, block=True, timeout=5.0) is True
    assert sleeps == [1.0]


def test_rate_limiter_manager_reuses_limiters_and_global_status(monkeypatch) -> None:
    monkeypatch.setattr(rate_limiter.time, "monotonic", lambda: 10.0)
    manager = rate_limiter.RateLimiterManager()

    first = manager.get_or_create("alpaca", rate=5.0, capacity=10)
    second = manager.get_or_create("alpaca", rate=1.0, capacity=1)

    assert first is second
    assert manager.get_all_status()["alpaca"]["capacity"] == 10

    monkeypatch.setattr(rate_limiter, "_rate_limiter_manager", manager)
    assert rate_limiter.get_rate_limiter("alpaca") is first
    assert "alpaca" in rate_limiter.get_all_rate_limiter_status()


def test_sliding_window_allows_expires_and_blocks_with_timeout(monkeypatch) -> None:
    limiter = rate_limiter.SlidingWindowRateLimiter(
        max_requests=1,
        window_seconds=60,
        name="news",
    )

    assert limiter.can_proceed() == (True, 0.0)
    blocked, wait_time = limiter.can_proceed()
    assert blocked is False
    assert wait_time > 0.0
    assert limiter.wait_if_needed(timeout=0.0) is False

    limiter.requests.clear()
    limiter.requests.append(datetime.now(UTC) - timedelta(seconds=61))
    assert limiter.can_proceed() == (True, 0.0)

    sleeps: list[float] = []
    monkeypatch.setattr(rate_limiter.time, "sleep", lambda seconds: sleeps.append(seconds))
    limiter.requests.clear()
    limiter.requests.append(datetime.now(UTC) - timedelta(seconds=59))
    limiter.max_requests = 2
    assert limiter.wait_if_needed(timeout=5.0) is True
    assert sleeps == []


def test_sliding_window_waits_until_capacity_available(monkeypatch) -> None:
    limiter = rate_limiter.SlidingWindowRateLimiter(max_requests=1, window_seconds=60)
    states = iter([(False, 0.25), (True, 0.0)])
    sleeps: list[float] = []
    monkeypatch.setattr(limiter, "can_proceed", lambda: next(states))
    monkeypatch.setattr(rate_limiter.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(rate_limiter.time, "sleep", lambda seconds: sleeps.append(seconds))

    assert limiter.wait_if_needed(timeout=None) is True
    assert sleeps == [0.25]
