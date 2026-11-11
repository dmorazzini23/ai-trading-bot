from datetime import date, timedelta
from types import SimpleNamespace

import pytest

from ai_trading.core import bot_engine


@pytest.fixture
def account(monkeypatch):
    bot_engine._init_metrics()
    holder = SimpleNamespace(equity=100_000.0)
    monkeypatch.setattr(bot_engine, "safe_alpaca_get_account", lambda _ctx: holder)
    monkeypatch.setattr(
        bot_engine, "_get_trading_config", lambda: SimpleNamespace(daily_loss_limit=0.05)
    )
    return holder


def _state_from_yesterday(start_equity: float) -> SimpleNamespace:
    yesterday = date.today() - timedelta(days=1)
    return SimpleNamespace(day_start_equity=(yesterday, start_equity), last_drawdown=0.0)


def test_day_after_large_loss_uses_tightened_limit(account):
    ctx = SimpleNamespace(api=SimpleNamespace())
    state = _state_from_yesterday(100_000.0)

    account.equity = 94_000.0  # yesterday lost 6%
    assert bot_engine.check_daily_loss(ctx, state) is False
    assert state.last_drawdown == pytest.approx(0.06)

    account.equity = 90_800.0  # down ~3.4% today: over 3%, under 5%
    assert bot_engine.check_daily_loss(ctx, state) is True


def test_day_after_small_loss_keeps_configured_limit(account):
    ctx = SimpleNamespace(api=SimpleNamespace())
    state = _state_from_yesterday(100_000.0)

    account.equity = 99_000.0  # yesterday lost 1%
    assert bot_engine.check_daily_loss(ctx, state) is False

    account.equity = 95_634.0  # down ~3.4% today
    assert bot_engine.check_daily_loss(ctx, state) is False
    account.equity = 94_000.0  # down ~5.05% today
    assert bot_engine.check_daily_loss(ctx, state) is True
