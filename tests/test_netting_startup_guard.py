"""Netting is the only decision path the service may start with."""

from __future__ import annotations

import logging
import sys
from types import SimpleNamespace

import pytest

from ai_trading import main


def test_startup_rejects_netting_disabled_outside_tests(monkeypatch, caplog):
    monkeypatch.setattr(main, "_is_test_mode", lambda: False)
    caplog.set_level(logging.CRITICAL)

    with pytest.raises(SystemExit) as excinfo:
        main._require_netting_pipeline(SimpleNamespace(netting_enabled=False))

    message = str(excinfo.value)
    assert "AI_TRADING_NETTING_ENABLED" in message
    assert "no longer supported" in message
    markers = [
        rec for rec in caplog.records if rec.getMessage() == "NETTING_DISABLED_UNSUPPORTED"
    ]
    assert len(markers) == 1
    assert markers[0].levelno == logging.CRITICAL
    assert markers[0].governing_flag == "AI_TRADING_NETTING_ENABLED"


def test_startup_accepts_netting_enabled_outside_tests(monkeypatch, caplog):
    monkeypatch.setattr(main, "_is_test_mode", lambda: False)
    caplog.set_level(logging.CRITICAL)

    main._require_netting_pipeline(SimpleNamespace(netting_enabled=True))

    assert not [
        rec for rec in caplog.records if rec.getMessage() == "NETTING_DISABLED_UNSUPPORTED"
    ]


def test_startup_guard_is_exempt_in_test_mode(monkeypatch):
    monkeypatch.setattr(main, "_is_test_mode", lambda: True)

    main._require_netting_pipeline(SimpleNamespace(netting_enabled=False))


def test_fail_fast_env_checks_resolved_trading_config(monkeypatch):
    seen: list[object] = []
    monkeypatch.setenv("AI_TRADING_NETTING_ENABLED", "0")
    monkeypatch.setattr(main, "_require_netting_pipeline", seen.append)

    main._fail_fast_env()

    assert len(seen) == 1
    assert seen[0].netting_enabled is False


def test_fail_fast_env_exits_when_netting_disabled_outside_tests(monkeypatch):
    monkeypatch.setattr(
        main,
        "_trading_config_from_env",
        lambda **_kwargs: SimpleNamespace(netting_enabled=False),
    )
    real_is_test_mode = main._is_test_mode
    calls = {"count": 0}

    def _test_mode_until_guard() -> bool:
        # The first call relaxes credential requirements; the guard asks second.
        calls["count"] += 1
        return real_is_test_mode() if calls["count"] == 1 else False

    monkeypatch.setattr(main, "_is_test_mode", _test_mode_until_guard)

    with pytest.raises(SystemExit, match="AI_TRADING_NETTING_ENABLED"):
        main._fail_fast_env()

    assert calls["count"] == 2


def _stub_run_trade_startup(monkeypatch, cli):
    monkeypatch.setattr(sys, "argv", ["ai_trading", "--paper", "--once", "--interval", "1"])
    monkeypatch.setattr(cli, "register_signal_handlers", lambda: None)
    monkeypatch.setattr(cli, "_set_execution_mode", lambda **_kwargs: None)
    monkeypatch.setattr(cli, "_validate_startup_config", lambda: None)
    monkeypatch.setattr("ai_trading.env.ensure_dotenv_loaded", lambda: None)
    loops: list[tuple[object, ...]] = []
    monkeypatch.setattr(cli, "_run_loop", lambda *args: loops.append(args))
    return loops


def test_run_trade_rejects_netting_disabled_before_any_cycle(monkeypatch):
    from ai_trading import __main__ as cli

    loops = _stub_run_trade_startup(monkeypatch, cli)
    monkeypatch.setattr(main, "_is_test_mode", lambda: False)
    monkeypatch.setattr(
        main,
        "preflight_import_health",
        lambda: pytest.fail("the netting check must run before the import preflight"),
    )
    monkeypatch.setenv("AI_TRADING_NETTING_ENABLED", "0")

    with pytest.raises(SystemExit, match="AI_TRADING_NETTING_ENABLED"):
        cli.run_trade()

    assert loops == []


def test_run_trade_starts_the_loop_when_netting_enabled(monkeypatch):
    from ai_trading import __main__ as cli

    loops = _stub_run_trade_startup(monkeypatch, cli)
    monkeypatch.setattr(main, "_is_test_mode", lambda: False)
    monkeypatch.setattr(main, "preflight_import_health", lambda: True)
    monkeypatch.setenv("AI_TRADING_NETTING_ENABLED", "1")

    cli.run_trade()

    assert len(loops) == 1
    assert loops[0][0] is main.run_cycle
