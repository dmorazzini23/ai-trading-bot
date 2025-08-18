from __future__ import annotations

import logging

from ai_trading.config.aliases import resolve_trading_mode
from ai_trading.logging import logger_once


def _clear_env(monkeypatch):
    for k in ("TRADING_MODE", "BOT_MODE", "bot_mode"):
        monkeypatch.delenv(k, raising=False)


def test_precedence(monkeypatch):
    _clear_env(monkeypatch)
    logger_once._emitted_keys.clear()
    assert resolve_trading_mode("balanced") == "balanced"
    monkeypatch.setenv("bot_mode", "aggressive")
    assert resolve_trading_mode("balanced") == "aggressive"
    monkeypatch.setenv("BOT_MODE", "moderate")
    assert resolve_trading_mode("balanced") == "moderate"
    monkeypatch.setenv("TRADING_MODE", "conservative")
    assert resolve_trading_mode("balanced") == "conservative"


def test_conflict_prefers_canonical(monkeypatch):
    _clear_env(monkeypatch)
    logger_once._emitted_keys.clear()
    monkeypatch.setenv("TRADING_MODE", "balanced")
    monkeypatch.setenv("BOT_MODE", "aggressive")
    assert resolve_trading_mode("moderate") == "balanced"


def test_deprecation_logged_once(monkeypatch, caplog):
    _clear_env(monkeypatch)
    logger_once._emitted_keys.clear()
    with caplog.at_level(logging.WARNING):
        monkeypatch.setenv("BOT_MODE", "aggressive")
        assert resolve_trading_mode("balanced") == "aggressive"
        assert caplog.records
    with caplog.at_level(logging.WARNING):
        caplog.clear()
        assert resolve_trading_mode("balanced") == "aggressive"
        assert not caplog.records

