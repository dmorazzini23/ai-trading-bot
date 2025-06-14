import sys
import types
from pathlib import Path

import pytest
from unittest.mock import patch, MagicMock

# Ensure project root is importable and stub heavy optional deps
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
mods = [
    "pandas",
    "numpy",
    "pandas_ta",
    "pandas_market_calendars",
    "pytz",
    "tzlocal",
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
    "alpaca.data.historical",
    "alpaca.data.models",
    "alpaca.data.requests",
    "alpaca.data.timeframe",
    "alpaca.common.exceptions",
    "dotenv",
    "finnhub",
    "joblib",
    "sklearn.ensemble",
    "sklearn.linear_model",
    "sklearn.decomposition",
    "pipeline",
    "metrics_logger",
    "prometheus_client",
    "pybreaker",
]
for m in mods:
    sys.modules.setdefault(m, types.ModuleType(m))
sys.modules["dotenv"].load_dotenv = lambda *a, **k: None
req_mod = types.ModuleType("requests")
sys.modules["requests"] = req_mod
exc_mod = types.ModuleType("requests.exceptions")
exc_mod.RequestException = Exception
exc_mod.HTTPError = Exception
req_mod.exceptions = exc_mod
req_mod.get = lambda *a, **k: None
sys.modules["requests.exceptions"] = exc_mod
sys.modules["urllib3"] = types.ModuleType("urllib3")
sys.modules["urllib3"].exceptions = types.SimpleNamespace(HTTPError=Exception)
sys.modules["alpaca_trade_api"].REST = object
sys.modules["alpaca_trade_api"].APIError = Exception
sys.modules["alpaca.common.exceptions"].APIError = Exception
sys.modules["alpaca.trading.client"].TradingClient = object
class _Req:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

sys.modules["alpaca.trading.requests"].LimitOrderRequest = _Req
sys.modules["alpaca.trading.requests"].MarketOrderRequest = _Req
sys.modules["alpaca.trading.requests"].GetOrdersRequest = _Req
sys.modules["alpaca.trading.enums"].OrderSide = types.SimpleNamespace(BUY="buy", SELL="sell")
sys.modules["alpaca.trading.enums"].TimeInForce = types.SimpleNamespace(DAY="day")
sys.modules["alpaca.trading.enums"].QueryOrderStatus = object
sys.modules["alpaca.trading.enums"].OrderStatus = object
sys.modules["alpaca.trading.models"].Order = object
sys.modules["alpaca.trading.stream"] = types.ModuleType("alpaca.trading.stream")
sys.modules["alpaca.trading.stream"].TradingStream = object
sys.modules["alpaca.data.models"].Quote = object
sys.modules["alpaca.data.requests"].StockLatestQuoteRequest = object
sys.modules["alpaca.data.requests"].StockBarsRequest = object
class _Client:
    def __init__(self, *a, **k):
        pass

sys.modules["alpaca.data.historical"].StockHistoricalDataClient = _Client
sys.modules["alpaca.data.timeframe"].TimeFrame = object
sys.modules["alpaca.data.timeframe"].TimeFrameUnit = object
class _FClient:
    def __init__(self, *a, **k):
        pass

sys.modules["finnhub"].Client = _FClient
sys.modules["finnhub"].FinnhubAPIException = Exception
sys.modules["bs4"].BeautifulSoup = lambda *a, **k: None
sys.modules["flask"].Flask = object
sys.modules["sklearn.ensemble"].RandomForestClassifier = object
sys.modules["sklearn.linear_model"].Ridge = object
sys.modules["sklearn.linear_model"].BayesianRidge = object
sys.modules["sklearn.decomposition"].PCA = object
sys.modules["prometheus_client"].start_http_server = lambda *a, **k: None
sys.modules["prometheus_client"].Counter = lambda *a, **k: None
sys.modules["prometheus_client"].Gauge = lambda *a, **k: None
sys.modules["prometheus_client"].Histogram = lambda *a, **k: None
sys.modules["pipeline"] = types.ModuleType("pipeline")
sys.modules["metrics_logger"] = types.ModuleType("metrics_logger")
sys.modules["joblib"] = types.ModuleType("joblib")
sys.modules["pybreaker"] = types.ModuleType("pybreaker")
sys.modules["pipeline"].model_pipeline = lambda *a, **k: None
sys.modules["metrics_logger"].log_metrics = lambda *a, **k: None
sys.modules["sentry_sdk"] = types.ModuleType("sentry_sdk")


def test_bot_main_normal(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "shadow")
    monkeypatch.setenv("APCA_API_KEY_ID", "k")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "s")
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setenv("FINNHUB_API_KEY", "testkey")
    monkeypatch.setattr("config.ALPACA_API_KEY", "k", raising=False)
    monkeypatch.setattr("config.ALPACA_SECRET_KEY", "s", raising=False)
    monkeypatch.setattr("config.FINNHUB_API_KEY", "testkey", raising=False)
    with patch("data_fetcher.get_minute_df", return_value=MagicMock()), \
         patch("alpaca_api.submit_order", return_value={"status": "mocked"}), \
         patch("signals.generate", return_value=1), \
         patch("risk_engine.calculate_position_size", return_value=10):
        import bot
        assert bot.main() is None or bot.main() is True


def test_bot_main_data_fetch_error(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "k")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "s")
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setenv("FINNHUB_API_KEY", "testkey")
    monkeypatch.setattr("config.ALPACA_API_KEY", "k", raising=False)
    monkeypatch.setattr("config.ALPACA_SECRET_KEY", "s", raising=False)
    monkeypatch.setattr("config.FINNHUB_API_KEY", "testkey", raising=False)
    with patch("data_fetcher.get_minute_df", side_effect=Exception("API error")):
        import bot
        with pytest.raises(Exception):
            bot.main()


def test_bot_main_signal_nan(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "k")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "s")
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setenv("FINNHUB_API_KEY", "testkey")
    monkeypatch.setattr("config.ALPACA_API_KEY", "k", raising=False)
    monkeypatch.setattr("config.ALPACA_SECRET_KEY", "s", raising=False)
    monkeypatch.setattr("config.FINNHUB_API_KEY", "testkey", raising=False)
    with patch("signals.generate", return_value=float('nan')), \
         patch("data_fetcher.get_minute_df", return_value=MagicMock()):
        import bot
        try:
            bot.main()
        except Exception:
            pytest.fail("Bot should handle NaN signal gracefully")


def test_trade_execution_api_timeout(monkeypatch):
    with patch("alpaca_api.submit_order", side_effect=TimeoutError("Timeout")), \
         patch("trade_execution.log_order") as mock_log:
        import trade_execution
        with pytest.raises(TimeoutError):
            trade_execution.place_order("AAPL", 5, "buy")
