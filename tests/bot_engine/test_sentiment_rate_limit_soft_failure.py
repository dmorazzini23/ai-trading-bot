from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def reset_sentiment_state():
    """Ensure sentiment circuit breaker starts from a clean slate."""

    from ai_trading.core import bot_engine as be

    be._SENTIMENT_CACHE.clear()
    be._SENTIMENT_CIRCUIT_BREAKER.update(
        {
            "failures": 0,
            "last_failure": 0,
            "state": "closed",
            "next_retry": 0,
            "opened_at": 0,
        }
    )
    yield
    # Tests expect a pristine circuit breaker for other cases as well.
    be._SENTIMENT_CACHE.clear()
    be._SENTIMENT_CIRCUIT_BREAKER.update(
        {
            "failures": 0,
            "last_failure": 0,
            "state": "closed",
            "next_retry": 0,
            "opened_at": 0,
        }
    )


def test_rate_limit_soft_failure_defers_provider_disable(monkeypatch):
    """Simulate repeated 429 responses and ensure escalation is deferred."""

    from ai_trading.core import bot_engine as be

    provider_calls: list[tuple[str, str, str | None]] = []

    def record_failure(provider: str, reason: str, error: str | None = None) -> None:
        provider_calls.append((provider, reason, error))

    monkeypatch.setattr(
        be.provider_monitor,
        "record_failure",
        record_failure,
        raising=False,
    )

    threshold = be.SENTIMENT_FAILURE_THRESHOLD

    for attempt in range(1, threshold):
        be._record_sentiment_failure("rate_limit")
        assert be._SENTIMENT_CIRCUIT_BREAKER["failures"] == attempt
        assert be._SENTIMENT_CIRCUIT_BREAKER["state"] == "closed"
        assert provider_calls == []

    be._record_sentiment_failure("rate_limit")
    assert be._SENTIMENT_CIRCUIT_BREAKER["failures"] == threshold
    assert be._SENTIMENT_CIRCUIT_BREAKER["state"] == "open"
    assert provider_calls == [("sentiment", "rate_limit", None)]
