import importlib

from ai_trading import settings
from ai_trading.utils import http, timing
from ai_trading.risk.engine import RiskEngine


def test_get_news_api_key_alias(monkeypatch):
    monkeypatch.delenv("NEWS_API_KEY", raising=False)
    monkeypatch.setenv("AI_TRADER_NEWS_API_KEY", "alt")
    importlib.reload(settings)
    assert settings.get_news_api_key() == "alt"


def test_http_timeout_alias(monkeypatch):
    monkeypatch.delenv("HTTP_TIMEOUT", raising=False)
    monkeypatch.setenv("HTTP_TIMEOUT_S", "20")
    importlib.reload(timing)
    assert timing.HTTP_TIMEOUT == 20.0


def test_http_workers_alias(monkeypatch):
    monkeypatch.delenv("HTTP_POOL_WORKERS", raising=False)
    monkeypatch.setenv("HTTP_MAX_WORKERS", "11")
    importlib.reload(http)
    assert http._pool_stats["workers"] == 11


def test_rebalance_interval_alias(monkeypatch):
    monkeypatch.setenv("AI_TRADER_REBALANCE_INTERVAL_MIN", "7")
    importlib.reload(settings)
    assert settings.get_rebalance_interval_min() == 7


def test_risk_engine_trade_slots():
    eng = RiskEngine()
    eng.max_trades = 2
    assert eng.acquire_trade_slot()
    assert eng.acquire_trade_slot()
    assert not eng.acquire_trade_slot()
    eng.release_trade_slot()
    assert eng.acquire_trade_slot()

