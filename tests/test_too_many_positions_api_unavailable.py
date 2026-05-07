import logging
from types import SimpleNamespace
from ai_trading.core import bot_engine


def test_too_many_positions_api_unavailable(caplog):
    ctx = SimpleNamespace(api=SimpleNamespace())
    with caplog.at_level(logging.WARNING):
        assert bot_engine.too_many_positions(ctx) is False
    assert "Positions API unavailable" in caplog.text


def test_live_position_guard_fails_closed_when_positions_unavailable(caplog):
    ctx = SimpleNamespace(api=SimpleNamespace(), execution_mode="live")
    with caplog.at_level(logging.WARNING):
        assert bot_engine.too_many_positions(ctx) is True
    assert "Positions API unavailable" in caplog.text


def test_live_loss_guards_fail_closed_when_account_unavailable(monkeypatch, caplog):
    ctx = SimpleNamespace(api=SimpleNamespace(), execution_mode="live")
    state = SimpleNamespace(day_start_equity=None, week_start_equity=None, last_drawdown=0.0)
    monkeypatch.setattr(bot_engine, "safe_alpaca_get_account", lambda _ctx: None)

    with caplog.at_level(logging.WARNING):
        assert bot_engine.check_daily_loss(ctx, state) is True
        assert bot_engine.check_weekly_loss(ctx, state) is True

    assert "Daily loss check skipped" in caplog.text
    assert "Weekly loss check skipped" in caplog.text


def test_live_buying_power_guard_returns_zero_when_account_unavailable():
    ctx = SimpleNamespace(api=SimpleNamespace(), execution_mode="live")

    qty, available = bot_engine._enforce_buying_power_limit(
        ctx,
        account=None,
        side="buy",
        price=100.0,
        qty=5,
    )

    assert qty == 0
    assert available is None
