import sys
import types
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

dummy = types.ModuleType("dummy")
mods = [
    "pandas_ta",
    "pandas_market_calendars",
    "requests",
    "urllib3",
    "bs4",
    "flask",
    "schedule",
    "portalocker",
    "alpaca",
    "alpaca.trading.client",
    "alpaca.trading.enums",
    "alpaca.trading.requests",
    "alpaca.trading.models",
    "alpaca_trade_api",
    "alpaca_trade_api.rest",
    "alpaca.data",
    "alpaca.data.stream",
    "alpaca.data.historical",
    "alpaca.data.models",
    "alpaca.data.requests",
    "alpaca.data.timeframe",
    "alpaca.common.exceptions",
    "sklearn.ensemble",
    "sklearn.linear_model",
    "sklearn.decomposition",
    "pipeline",
    "metrics_logger",
    "prometheus_client",
    "finnhub",
    "joblib",
    "pybreaker",
    "trade_execution",
    "capital_scaling",
    "strategy_allocator",
    "risk_engine",
    "strategies",
    "strategies.momentum",
    "strategies.mean_reversion",
]
for name in mods:
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)
sys.modules["pipeline"].model_pipeline = lambda *a, **k: None
sys.modules["alpaca.data"].Stream = object
sys.modules["alpaca.data.stream"].Stream = object

sys.modules["flask"].Flask = object
sys.modules["requests"].get = lambda *a, **k: None
sys.modules["requests"].exceptions = types.SimpleNamespace(RequestException=Exception)
sys.modules["urllib3"] = types.ModuleType("urllib3")
sys.modules["urllib3"].exceptions = types.SimpleNamespace(HTTPError=Exception)
sys.modules["alpaca_trade_api"].REST = object
sys.modules["alpaca_trade_api"].APIError = Exception
sys.modules["alpaca.common.exceptions"].APIError = Exception
sys.modules["sklearn.ensemble"].RandomForestClassifier = object
sys.modules["sklearn.linear_model"].Ridge = object
sys.modules["sklearn.linear_model"].BayesianRidge = object
sys.modules["sklearn.decomposition"].PCA = object
sys.modules["joblib"] = types.ModuleType("joblib")


class _FakeREST:
    def __init__(self, *a, **k):
        pass


sys.modules["alpaca_trade_api.rest"].REST = _FakeREST
sys.modules["alpaca_trade_api.rest"].APIError = Exception
sys.modules["alpaca.trading.client"].TradingClient = object
sys.modules["alpaca.trading.enums"].OrderSide = object
sys.modules["alpaca.trading.enums"].TimeInForce = object
sys.modules["alpaca.trading.enums"].QueryOrderStatus = object
sys.modules["alpaca.trading.requests"].GetOrdersRequest = object
sys.modules["alpaca.trading.requests"].MarketOrderRequest = object
sys.modules["alpaca.trading.requests"].LimitOrderRequest = object
sys.modules["alpaca.trading.models"].Order = object
sys.modules["alpaca.data.historical"].StockHistoricalDataClient = object
sys.modules["alpaca.data.models"].Quote = object
sys.modules["alpaca.data.requests"].StockBarsRequest = object
sys.modules["alpaca.data.requests"].StockLatestQuoteRequest = object
sys.modules["alpaca.data.timeframe"].TimeFrame = object

sys.modules["bs4"].BeautifulSoup = lambda *a, **k: None
sys.modules["prometheus_client"].start_http_server = lambda *a, **k: None
sys.modules["prometheus_client"].Counter = lambda *a, **k: None
sys.modules["prometheus_client"].Gauge = lambda *a, **k: None
sys.modules["prometheus_client"].Histogram = lambda *a, **k: None

bot = pytest.importorskip("bot")


def test_screen_candidates_empty(monkeypatch):
    monkeypatch.setattr(bot, "load_tickers", lambda path=bot.TICKERS_FILE: ["AAA"])
    monkeypatch.setattr(bot, "screen_universe", lambda candidates, ctx: [])
    assert bot.screen_candidates() == []
