from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from ai_trading.portfolio import sizing


@pytest.fixture(autouse=True)
def _reset_equity_cache() -> None:
    sizing._equity_cache.ts = None
    sizing._equity_cache.equity = None
    yield
    sizing._equity_cache.ts = None
    sizing._equity_cache.equity = None


def test_fetch_equity_uses_cache_within_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    sizing._equity_cache.ts = now - timedelta(seconds=30)
    sizing._equity_cache.equity = 4321.0
    monkeypatch.setattr(sizing, "_now", lambda: now)

    class _Api:
        def get_account(self):  # pragma: no cover - should not be called
            raise AssertionError("cache should have been used")

    ctx = SimpleNamespace(api=_Api())
    assert sizing._fetch_equity(ctx, ttl_seconds=60) == 4321.0


def test_fetch_equity_refreshes_and_sets_paper_flag() -> None:
    class _Api:
        def get_account(self):
            return SimpleNamespace(equity="123.45")

    api = _Api()
    ctx = SimpleNamespace(api=api, alpaca_base_url="https://paper-api.alpaca.markets")
    equity = sizing._fetch_equity(ctx, force_refresh=True)

    assert equity == pytest.approx(123.45)
    assert getattr(api, "paper") is True


def test_fetch_equity_error_returns_zero_and_updates_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    monkeypatch.setattr(sizing, "_now", lambda: now)

    class _Api:
        def get_account(self):
            raise TypeError("boom")

    ctx = SimpleNamespace(api=_Api())
    equity = sizing._fetch_equity(ctx, force_refresh=True)

    assert equity == 0.0
    assert sizing._equity_cache.equity == 0.0
    assert sizing._equity_cache.ts == now


def test_volatility_targeting_applies_limits_and_normalizes() -> None:
    sizer = sizing.VolatilityTargetingSizer(min_weight=0.05, max_weight=0.6)
    adjusted = sizer._apply_position_limits({"A": 0.01, "B": 0.5, "C": 0.8})

    assert "A" not in adjusted
    assert adjusted["B"] == pytest.approx(0.45454545)
    assert adjusted["C"] == pytest.approx(0.54545454)
    assert sum(adjusted.values()) == pytest.approx(1.0)


def test_turnover_penalty_dampens_to_configured_limit() -> None:
    sizer = sizing.TurnoverPenaltySizer(max_turnover=0.25)
    current = {"A": 1.0}
    proposed = {"B": 1.0}

    adjusted = sizer.apply_turnover_penalty(proposed, current)

    assert adjusted["A"] == pytest.approx(0.75)
    assert adjusted["B"] == pytest.approx(0.25)
    assert sizer._calculate_turnover(adjusted, current) <= 0.250000001


def test_turnover_penalty_keeps_weights_when_already_within_limit() -> None:
    sizer = sizing.TurnoverPenaltySizer(max_turnover=0.5)
    current = {"A": 0.6, "B": 0.4}
    proposed = {"A": 0.55, "B": 0.45}

    adjusted = sizer.apply_turnover_penalty(proposed, current)

    assert adjusted == proposed
    assert len(sizer.position_history) == 1


def test_get_historical_turnover_returns_series() -> None:
    sizer = sizing.TurnoverPenaltySizer()
    sizer._update_position_history({"A": 1.0})
    sizer._update_position_history({"A": 0.5, "B": 0.5})
    sizer._update_position_history({"A": 0.25, "B": 0.75})

    turnover = sizer.get_historical_turnover()
    assert turnover == [0.5, 0.25]
