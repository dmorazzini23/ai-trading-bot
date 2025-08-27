import importlib
import pytest

from ai_trading import settings
from ai_trading.risk.engine import RiskEngine

try:
    from ai_trading.utils import timing  # type: ignore
except Exception:  # AI-AGENT-REF: timing module optional in tests
    timing = None


def test_get_news_api_key_env(monkeypatch):
    monkeypatch.delenv("NEWS_API_KEY", raising=False)
    monkeypatch.setenv("AI_TRADING_NEWS_API_KEY", "alt")
    importlib.reload(settings)
    assert settings.get_news_api_key() == "alt"


@pytest.mark.skipif(timing is None, reason="timing helpers not available")
def test_http_timeout_env(monkeypatch):
    monkeypatch.setenv("HTTP_TIMEOUT", "20")
    importlib.reload(timing)
    assert timing.HTTP_TIMEOUT == 20.0


def test_rebalance_interval_env(monkeypatch):
    monkeypatch.setenv("AI_TRADING_REBALANCE_INTERVAL_MIN", "7")
    importlib.reload(settings)
    assert settings.get_rebalance_interval_min() == 7


def test_risk_engine_trade_slots(monkeypatch):
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    eng = RiskEngine()
    eng.max_trades = 2
    assert eng.acquire_trade_slot()
    assert eng.acquire_trade_slot()
    assert not eng.acquire_trade_slot()
    eng.release_trade_slot()
    assert eng.acquire_trade_slot()

