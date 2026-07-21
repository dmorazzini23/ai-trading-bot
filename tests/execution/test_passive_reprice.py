from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ai_trading.execution.passive_reprice import (
    deterministic_passive_reprice_id,
    validate_passive_reprice_quote,
)


def test_deterministic_passive_reprice_id_is_stable_and_bounded() -> None:
    root = "root/decision:" + ("x" * 80)

    first = deterministic_passive_reprice_id(root, generation=2)
    second = deterministic_passive_reprice_id(root, generation=2)

    assert first == second
    assert first.endswith("-pr2")
    assert len(first) <= 48
    assert "/" not in first
    assert ":" not in first


@pytest.mark.parametrize(
    ("side", "expected"),
    [("buy", 100.01), ("sell", 100.04)],
)
def test_passive_reprice_stays_at_current_best_quote(
    side: str,
    expected: float,
) -> None:
    now = datetime(2026, 7, 21, 15, 0, tzinfo=UTC)

    decision = validate_passive_reprice_quote(
        side=side,
        bid=100.019,
        ask=100.031,
        quote_ts=now - timedelta(milliseconds=125),
        now=now,
        tick_size=0.01,
        max_quote_age_ms=500,
        max_spread_bps=5.0,
    )

    assert decision.allowed is True
    assert decision.reason == "ok"
    assert decision.limit_price == expected
    assert decision.quote_age_ms == pytest.approx(125.0)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"bid": 100.0, "ask": 100.0}, "quote_locked_or_crossed"),
        ({"quote_ts": None}, "quote_timestamp_missing"),
        ({"max_spread_bps": 1.0}, "spread_above_max"),
    ],
)
def test_passive_reprice_rejects_unsafe_quotes(
    overrides: dict[str, object],
    reason: str,
) -> None:
    now = datetime(2026, 7, 21, 15, 0, tzinfo=UTC)
    kwargs: dict[str, object] = {
        "side": "buy",
        "bid": 100.0,
        "ask": 100.02,
        "quote_ts": now,
        "now": now,
        "tick_size": 0.01,
        "max_quote_age_ms": 500,
        "max_spread_bps": 5.0,
    }
    kwargs.update(overrides)

    decision = validate_passive_reprice_quote(**kwargs)  # type: ignore[arg-type]

    assert decision.allowed is False
    assert decision.reason == reason
    assert decision.limit_price is None


def test_passive_reprice_rejects_stale_quote() -> None:
    now = datetime(2026, 7, 21, 15, 0, tzinfo=UTC)

    decision = validate_passive_reprice_quote(
        side="sell",
        bid=100.0,
        ask=100.02,
        quote_ts=now - timedelta(milliseconds=501),
        now=now,
        max_quote_age_ms=500,
        max_spread_bps=5.0,
    )

    assert decision.allowed is False
    assert decision.reason == "quote_age_above_max"


@pytest.mark.parametrize("generation", [0, -1])
def test_deterministic_passive_reprice_id_rejects_invalid_generation(
    generation: int,
) -> None:
    with pytest.raises(ValueError, match="generation must be positive"):
        deterministic_passive_reprice_id("root", generation=generation)
