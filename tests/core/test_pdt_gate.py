from types import SimpleNamespace

import pytest

import ai_trading.core.bot_engine as bot_engine


def _runtime_with_account(
    monkeypatch: pytest.MonkeyPatch,
    enforce_daytrade_limit: bool = False,
    **account_fields,
):
    runtime = SimpleNamespace(api=object(), enforce_daytrade_limit=enforce_daytrade_limit)
    monkeypatch.setattr(bot_engine, "ensure_alpaca_attached", lambda _ctx: None)
    monkeypatch.setattr(
        bot_engine,
        "safe_alpaca_get_account",
        lambda _ctx: SimpleNamespace(**account_fields),
    )
    return runtime


def test_pdt_allows_when_equity_ok_and_dtbp_positive(monkeypatch: pytest.MonkeyPatch):
    runtime = _runtime_with_account(
        monkeypatch,
        pattern_day_trader=True,
        equity="50000",
        daytrading_buying_power="100000",
        trading_blocked=False,
        account_blocked=False,
    )
    assert bot_engine.check_pdt_rule(runtime) is False


def test_pdt_blocks_when_broker_flag_set(monkeypatch: pytest.MonkeyPatch):
    runtime = _runtime_with_account(
        monkeypatch,
        pattern_day_trader=True,
        equity="80000",
        daytrading_buying_power="0",
        trading_blocked=True,
        account_blocked=False,
    )
    assert bot_engine.check_pdt_rule(runtime) is True
    context = getattr(runtime, "_pdt_last_context")
    assert context["pdt_equity_ok"] is True
    assert context.get("block_reason") == "broker_blocked"


def test_pdt_blocks_when_equity_below_threshold(monkeypatch: pytest.MonkeyPatch):
    low_equity = bot_engine.PDT_EQUITY_THRESHOLD - 1
    runtime = _runtime_with_account(
        monkeypatch,
        pattern_day_trader=True,
        equity=str(low_equity),
        daytrading_buying_power="100000",
        trading_blocked=False,
        account_blocked=False,
    )
    assert bot_engine.check_pdt_rule(runtime) is True
    context = getattr(runtime, "_pdt_last_context")
    assert context["pdt_equity_ok"] is False
    assert context.get("block_reason") == "equity_below_threshold"


def test_pdt_blocks_when_dtbp_exhausted(monkeypatch: pytest.MonkeyPatch):
    runtime = _runtime_with_account(
        monkeypatch,
        pattern_day_trader=True,
        equity=str(bot_engine.PDT_EQUITY_THRESHOLD + 1000),
        daytrading_buying_power="0",
        trading_blocked=False,
        account_blocked=False,
    )
    assert bot_engine.check_pdt_rule(runtime) is True
    context = getattr(runtime, "_pdt_last_context")
    assert context["pdt_equity_ok"] is True
    assert context.get("block_reason") == "dtbp_exhausted"


def test_pdt_blocks_when_daytrade_limit_exhausted(monkeypatch: pytest.MonkeyPatch):
    runtime = _runtime_with_account(
        monkeypatch,
        enforce_daytrade_limit=True,
        pattern_day_trader=True,
        equity=str(bot_engine.PDT_EQUITY_THRESHOLD + 1000),
        daytrade_count=4,
        daytrade_limit=4,
        daytrading_buying_power="100000",
        trading_blocked=False,
        account_blocked=False,
    )
    assert bot_engine.check_pdt_rule(runtime) is False
    context = getattr(runtime, "_pdt_last_context")
    assert context["pdt_equity_ok"] is True
    assert context.get("block_reason") is None
