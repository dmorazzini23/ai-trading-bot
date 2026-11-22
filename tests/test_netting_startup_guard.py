"""Netting is the only decision path; the retired netting flag selects nothing."""

from __future__ import annotations

import sys

import pytest

from ai_trading import main


def _stub_run_trade_startup(monkeypatch, cli):
    monkeypatch.setattr(sys, "argv", ["ai_trading", "--paper", "--once", "--interval", "1"])
    monkeypatch.setattr(cli, "register_signal_handlers", lambda: None)
    monkeypatch.setattr(cli, "_set_execution_mode", lambda **_kwargs: None)
    monkeypatch.setattr(cli, "_validate_startup_config", lambda: None)
    monkeypatch.setattr("ai_trading.env.ensure_dotenv_loaded", lambda: None)
    loops: list[tuple[object, ...]] = []
    monkeypatch.setattr(cli, "_run_loop", lambda *args: loops.append(args))
    return loops


def test_trading_config_has_no_netting_switch(monkeypatch):
    from ai_trading.config.runtime import SPEC_BY_ENV, SPEC_BY_FIELD, TradingConfig

    monkeypatch.setenv("AI_TRADING_NETTING_ENABLED", "0")

    assert "netting_enabled" not in SPEC_BY_FIELD
    assert "AI_TRADING_NETTING_ENABLED" not in SPEC_BY_ENV
    assert not hasattr(TradingConfig.from_env(allow_missing_drawdown=True), "netting_enabled")


@pytest.mark.parametrize("flag", ["0", "1"])
def test_run_trade_starts_the_loop_whatever_the_retired_netting_flag_says(monkeypatch, flag):
    from ai_trading import __main__ as cli

    loops = _stub_run_trade_startup(monkeypatch, cli)
    monkeypatch.setattr(main, "_is_test_mode", lambda: False)
    monkeypatch.setattr(main, "preflight_import_health", lambda: True)
    monkeypatch.setenv("AI_TRADING_NETTING_ENABLED", flag)

    cli.run_trade()

    assert len(loops) == 1
    assert loops[0][0] is main.run_cycle
