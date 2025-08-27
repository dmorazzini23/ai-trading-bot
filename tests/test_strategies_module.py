import os
import sys
import types

# AI-AGENT-REF: minimal stubs for heavy optional deps
os.environ.setdefault("PYTEST_RUNNING", "1")
sys.modules.setdefault(
    "pandas_market_calendars",
    types.SimpleNamespace(get_calendar=lambda *a, **k: None),
)
sys.modules.setdefault("alpaca", types.ModuleType("alpaca"))
sys.modules.setdefault("alpaca.trading", types.ModuleType("alpaca.trading"))
trading_mod = types.ModuleType("alpaca.trading.client")
trading_mod.TradingClient = object
trading_mod.APIError = type("APIError", (Exception,), {})
sys.modules["alpaca.trading.client"] = trading_mod
data_mod = types.ModuleType("alpaca.data")
tf_mod = types.ModuleType("alpaca.data.timeframe")
tf_mod.TimeFrame = type("TimeFrame", (object,), {})
tf_mod.TimeFrameUnit = type("TimeFrameUnit", (object,), {})
req_mod = types.ModuleType("alpaca.data.requests")
req_mod.StockBarsRequest = type("StockBarsRequest", (object,), {})
sys.modules["alpaca.data"] = data_mod
sys.modules["alpaca.data.timeframe"] = tf_mod
sys.modules["alpaca.data.requests"] = req_mod
data_mod.TimeFrame = tf_mod.TimeFrame
data_mod.StockBarsRequest = req_mod.StockBarsRequest

tzlocal_mod = types.ModuleType("tzlocal")
tzlocal_mod.get_localzone = lambda: None
sys.modules["tzlocal"] = tzlocal_mod

# Stub internal modules pulled in by bot_engine imports we don't exercise
df_stub = types.ModuleType("ai_trading.data.fetch")
df_stub.get_bars = df_stub.get_bars_batch = lambda *a, **k: []
df_stub.get_minute_df = lambda *a, **k: None
df_stub.DataFetchError = Exception
df_stub.get_cached_minute_timestamp = lambda *a, **k: 0
df_stub.last_minute_bar_age_seconds = lambda *a, **k: 0
data_pkg = types.ModuleType("ai_trading.data")
data_pkg.fetch = df_stub
sys.modules["ai_trading.data"] = data_pkg
sys.modules["ai_trading.data.fetch"] = df_stub

market_pkg = types.ModuleType("ai_trading.market")
cal_stub = types.ModuleType("ai_trading.market.calendars")
cal_stub.ensure_final_bar = lambda *a, **k: None
sys.modules["ai_trading.market"] = market_pkg
sys.modules["ai_trading.market.calendars"] = cal_stub

risk_pkg = types.ModuleType("ai_trading.risk")
cb_stub = types.ModuleType("ai_trading.risk.circuit_breakers")
cb_stub.DrawdownCircuitBreaker = object
sys.modules["ai_trading.risk"] = risk_pkg
sys.modules["ai_trading.risk.circuit_breakers"] = cb_stub
adaptive_stub = types.ModuleType("ai_trading.risk.adaptive_sizing")
adaptive_stub.AdaptivePositionSizer = object
sys.modules["ai_trading.risk.adaptive_sizing"] = adaptive_stub

pandas_ta_stub = types.ModuleType("pandas_ta")
pandas_ta_stub._bind_known_methods = lambda: None
sys.modules["pandas_ta"] = pandas_ta_stub

rebalancer_stub = types.ModuleType("ai_trading.rebalancer")
rebalancer_stub.maybe_rebalance = lambda *a, **k: None
rebalancer_stub.rebalance_if_needed = lambda *a, **k: None
sys.modules["ai_trading.rebalancer"] = rebalancer_stub

pipeline_stub = types.ModuleType("ai_trading.pipeline")
pipeline_stub.model_pipeline = lambda *a, **k: None
sys.modules["ai_trading.pipeline"] = pipeline_stub

finnhub_stub = types.ModuleType("finnhub")
finnhub_stub.FinnhubAPIException = type("FinnhubAPIException", (Exception,), {})
sys.modules["finnhub"] = finnhub_stub

from ai_trading.config import settings as settings_module
from ai_trading.core.bot_engine import get_strategies


def _prep_settings(strategies):
    """Reset cached settings and optionally set strategies_wanted."""  # AI-AGENT-REF: test helper
    settings_module.get_settings.cache_clear()
    S = settings_module.get_settings()
    if strategies is not None:
        object.__setattr__(S, "strategies_wanted", strategies)
    return S


def test_get_strategies_non_empty_for_empty_settings(monkeypatch):
    _prep_settings([])
    monkeypatch.delenv("STRATEGIES", raising=False)
    assert get_strategies()


def test_get_strategies_non_empty_for_unknown_settings(monkeypatch):
    _prep_settings(["unknown"])
    monkeypatch.delenv("STRATEGIES", raising=False)
    assert get_strategies()


def test_get_strategies_non_empty_when_env_unset(monkeypatch):
    _prep_settings(None)
    monkeypatch.delenv("STRATEGIES", raising=False)
    assert get_strategies()

