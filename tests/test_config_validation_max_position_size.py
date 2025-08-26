import os
import importlib
from contextlib import contextmanager
import pytest


@contextmanager
def _temp_env(k, v):  # AI-AGENT-REF: helper to toggle env vars
    old = os.environ.get(k)
    if v is None and k in os.environ:
        del os.environ[k]
    elif v is not None:
        os.environ[k] = str(v)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = old


def test_env_override_precedence(monkeypatch):  # AI-AGENT-REF: env override respected
    with _temp_env("AI_TRADING_MAX_POSITION_SIZE", "1234.5"):
        ps = importlib.import_module("ai_trading.position_sizing")
        val, src = ps._resolve_max_position_size(0.0, 0.1, None)
        assert val == 1234.5
        assert src == "env_override"


def test_no_mutation_of_settings(monkeypatch):  # AI-AGENT-REF: ensure env fallback
    import ai_trading.main as m

    class Dummy:
        trading_mode = "balanced"
        capital_cap = 0.5
        dollar_risk_limit = 0.4

    m.logger = getattr(m, "logger", None) or importlib.import_module("logging").getLogger(__name__)

    class CfgDummy:  # AI-AGENT-REF: minimal cfg for validation
        alpaca_base_url = "paper"
        paper = True

    with _temp_env("AI_TRADING_MAX_POSITION_SIZE", None):
        d = Dummy()
        m._validate_runtime_config(cfg=CfgDummy(), tcfg=d)
        assert not hasattr(d, "max_position_size")
        assert os.environ.get("AI_TRADING_MAX_POSITION_SIZE") is not None


def test_negative_max_position_size_rejected():  # AI-AGENT-REF: reject nonpositive
    with _temp_env("MAX_POSITION_SIZE", "-1"):
        from ai_trading.config.management import TradingConfig

        with pytest.raises(ValueError, match="MAX_POSITION_SIZE must be positive"):
            TradingConfig.from_env()


def test_negative_env_override_rejected():  # AI-AGENT-REF: reject override
    with _temp_env("AI_TRADING_MAX_POSITION_SIZE", "-5"):
        ps = importlib.import_module("ai_trading.position_sizing")
        with pytest.raises(ValueError, match="AI_TRADING_MAX_POSITION_SIZE must be positive"):
            ps._resolve_max_position_size(0.0, 0.1, None)

