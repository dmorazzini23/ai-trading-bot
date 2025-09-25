import importlib
import logging
import os
import sys

import pytest

from ai_trading.config.legacy_env import LEGACY_ALPACA_ENV_MAPPING
from ai_trading.config.management import TradingConfig


def test_ai_trading_aliases_backfill_when_canonical_missing(monkeypatch):
    for key in (
        "BUY_THRESHOLD",
        "CONF_THRESHOLD",
        "MAX_DRAWDOWN_THRESHOLD",
        "MAX_POSITION_SIZE",
        "POSITION_SIZE_MIN_USD",
        "CONFIDENCE_LEVEL",
        "KELLY_FRACTION_MAX",
        "MIN_SAMPLE_SIZE",
    ):
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv("AI_TRADING_BUY_THRESHOLD", "0.4")
    monkeypatch.setenv("AI_TRADING_CONF_THRESHOLD", "0.8")
    monkeypatch.setenv("AI_TRADING_MAX_DRAWDOWN_THRESHOLD", "0.08")
    monkeypatch.setenv("AI_TRADING_MAX_POSITION_SIZE", "9000")
    monkeypatch.setenv("AI_TRADING_POSITION_SIZE_MIN_USD", "25")
    monkeypatch.setenv("AI_TRADING_CONFIDENCE_LEVEL", "0.85")
    monkeypatch.setenv("AI_TRADING_KELLY_FRACTION_MAX", "0.20")
    monkeypatch.setenv("AI_TRADING_MIN_SAMPLE_SIZE", "12")

    cfg = TradingConfig.from_env({})

    assert cfg.buy_threshold == 0.4
    assert cfg.conf_threshold == 0.8
    assert cfg.max_drawdown_threshold == 0.08
    assert cfg.max_position_size == 9000
    assert cfg.position_size_min_usd == 25
    assert cfg.confidence_level == 0.85
    assert cfg.kelly_fraction_max == 0.20
    assert cfg.min_sample_size == 12


def test_ai_trading_alias_respects_existing_canonical(monkeypatch):
    monkeypatch.setenv("MAX_DRAWDOWN_THRESHOLD", "0.2")
    monkeypatch.setenv("MAX_POSITION_SIZE", "5000")
    monkeypatch.setenv("AI_TRADING_MAX_POSITION_SIZE", "9000")
    monkeypatch.setenv("CONF_THRESHOLD", "0.6")
    monkeypatch.setenv("AI_TRADING_CONF_THRESHOLD", "0.9")

    cfg = TradingConfig.from_env({})

    assert cfg.max_position_size == 5000
    assert cfg.conf_threshold == 0.6


def test_legacy_alias_backfills_only_when_missing(monkeypatch):
    monkeypatch.delenv("DOLLAR_RISK_LIMIT", raising=False)
    monkeypatch.setenv("DAILY_LOSS_LIMIT", "0.25")

    cfg = TradingConfig.from_env({})

    assert cfg.dollar_risk_limit == 0.25


def test_legacy_alias_does_not_override_existing_canonical(monkeypatch):
    monkeypatch.setenv("DOLLAR_RISK_LIMIT", "0.42")
    monkeypatch.setenv("DAILY_LOSS_LIMIT", "0.25")

    cfg = TradingConfig.from_env({})

    assert cfg.dollar_risk_limit == 0.42


def test_runtime_import_backfills_legacy_alpaca_env(monkeypatch, caplog):
    sys.modules.pop("ai_trading.config.runtime", None)
    for key, canonical in LEGACY_ALPACA_ENV_MAPPING.items():
        monkeypatch.setenv(key, f"legacy-{key.lower()}")
        monkeypatch.delenv(canonical, raising=False)

    with caplog.at_level(logging.INFO):
        importlib.import_module("ai_trading.config.runtime")

    assert os.getenv("ALPACA_API_KEY") == "legacy-apca_api_key_id"
    assert os.getenv("ALPACA_SECRET_KEY") == "legacy-apca_api_secret_key"
    assert any(record.message == "ALPACA_LEGACY_ENV_BACKFILLED" for record in caplog.records)
    assert any(record.message == "ALPACA_LEGACY_ENV_CONFLICT" for record in caplog.records)

    # Subsequent imports should not duplicate logs or alter values.
    sys.modules.pop("ai_trading.config.runtime", None)
    caplog.clear()
    with caplog.at_level(logging.INFO):
        importlib.import_module("ai_trading.config.runtime")

    assert os.getenv("ALPACA_API_KEY") == "legacy-apca_api_key_id"
    assert os.getenv("ALPACA_SECRET_KEY") == "legacy-apca_api_secret_key"
    assert all(record.message != "ALPACA_LEGACY_ENV_BACKFILLED" for record in caplog.records)
