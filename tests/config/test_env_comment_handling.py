import importlib
import sys

import pytest


def _reload_config(monkeypatch, **env):
    module_name = "ai_trading.config"
    sys.modules.pop(module_name, None)
    for key in [
        "MAX_DRAWDOWN_THRESHOLD",
        "AI_TRADING_MAX_DRAWDOWN_THRESHOLD",
        "TRADING_MODE",
        "KELLY_FRACTION",
        "CONF_THRESHOLD",
        "MAX_POSITION_SIZE",
    ]:
        monkeypatch.delenv(key, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    return importlib.import_module(module_name)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    sys.modules.pop("ai_trading.config", None)


def test_comment_stripped_from_float_env(monkeypatch):
    cfg = _reload_config(monkeypatch, MAX_DRAWDOWN_THRESHOLD="0.08  # comment")
    assert cfg.get_max_drawdown_threshold() == pytest.approx(0.08)

