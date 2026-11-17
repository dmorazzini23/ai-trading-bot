from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Iterator, cast

import numpy as np
import pandas as pd
import pytest

from ai_trading.portfolio import sizing


@pytest.fixture(autouse=True)
def _reset_equity_cache() -> Iterator[None]:
    sizing._equity_cache.ts = None
    sizing._equity_cache.equity = None
    yield
    sizing._equity_cache.ts = None
    sizing._equity_cache.equity = None


def _series(start: float, step: float, n: int = 80) -> pd.Series:
    return pd.Series([start + step * i for i in range(n)], dtype=float)


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




















def test_position_size_callable_for_bot_engine_hook() -> None:
    position_size = getattr(sizing, "position_size", None)

    assert callable(position_size)
    assert position_size(
        "AAPL",
        0.5,
        100_000.0,
        "moderate",
        current_price=100.0,
    ) == 25
























