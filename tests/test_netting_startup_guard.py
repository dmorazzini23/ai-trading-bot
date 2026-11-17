"""Netting is the only decision path the service may start with."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from ai_trading import main
from ai_trading.core import bot_engine


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


def test_startup_allows_explicit_legacy_opt_in_under_tests(monkeypatch):
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


def test_netting_pipeline_enabled_by_default_without_runtime_cfg(monkeypatch):
    monkeypatch.delenv("AI_TRADING_NETTING_ENABLED", raising=False)

    assert bot_engine._pytest_running() is True
    assert bot_engine._netting_pipeline_enabled(SimpleNamespace(cfg=None)) is True


def test_netting_pipeline_follows_explicit_legacy_opt_in(monkeypatch):
    monkeypatch.setenv("AI_TRADING_NETTING_ENABLED", "0")

    assert bot_engine._netting_pipeline_enabled(SimpleNamespace(cfg=None)) is False
