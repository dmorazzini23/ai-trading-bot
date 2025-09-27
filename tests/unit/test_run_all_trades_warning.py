"""Ensure run_all_trades_worker does not warn for a valid API client."""

from __future__ import annotations

import types
import sys
from unittest.mock import Mock

import pytest

sklearn_stub = types.ModuleType("sklearn")
ensemble_stub = types.ModuleType("sklearn.ensemble")

class _GB:  # noqa: D401 - minimal placeholder
    pass


class _RF:  # noqa: D401 - minimal placeholder
    pass


ensemble_stub.GradientBoostingClassifier = _GB
ensemble_stub.RandomForestClassifier = _RF
sys.modules.setdefault("sklearn", sklearn_stub)
sys.modules.setdefault("sklearn.ensemble", ensemble_stub)
metrics_stub = types.ModuleType("sklearn.metrics")
metrics_stub.accuracy_score = lambda *a, **k: 0.0
sys.modules.setdefault("sklearn.metrics", metrics_stub)
model_selection_stub = types.ModuleType("sklearn.model_selection")
model_selection_stub.train_test_split = lambda *a, **k: ([], [])
sys.modules.setdefault("sklearn.model_selection", model_selection_stub)
preproc_stub = types.ModuleType("sklearn.preprocessing")
preproc_stub.StandardScaler = type("StandardScaler", (), {})
sys.modules.setdefault("sklearn.preprocessing", preproc_stub)

numpy_stub = types.ModuleType("numpy")
numpy_stub.nan = float("nan")
numpy_stub.NaN = float("nan")
numpy_stub.isfinite = lambda *_a, **_k: True
numpy_stub.asarray = lambda value, *a, **k: value
numpy_stub.array = lambda value, *a, **k: value
numpy_stub.stack = lambda seq, *a, **k: list(seq)
numpy_stub.std = lambda *_a, **_k: 0.0
numpy_stub.mean = lambda *_a, **_k: 0.0
numpy_stub.clip = lambda arr, *a, **k: arr
numpy_stub.argmax = lambda *_a, **_k: 0
numpy_stub.where = lambda cond, x, y: x if cond else y
numpy_stub.float32 = float
numpy_stub.float64 = float
numpy_stub.eye = lambda n, dtype=None: [
    [1 if i == j else 0 for j in range(n)] for i in range(n)
]
numpy_stub.zeros = lambda shape, dtype=None: [0] * shape if isinstance(shape, int) else [
    [0 for _ in range(shape[1])] for _ in range(shape[0])
]
numpy_stub.ones = lambda shape, dtype=None: [1] * shape if isinstance(shape, int) else [
    [1 for _ in range(shape[1])] for _ in range(shape[0])
]
numpy_stub.random = types.SimpleNamespace(seed=lambda *_a, **_k: None)
numpy_stub.polyfit = lambda *_a, **_k: [0.0]
numpy_stub.where = lambda cond, x, y: x if cond else y
numpy_stub.isnan = lambda *_a, **_k: False
numpy_stub.isscalar = lambda obj: not isinstance(obj, (list, tuple, dict, set))
numpy_stub.bool_ = bool
sys.modules.setdefault("numpy", numpy_stub)

portalocker_stub = types.ModuleType("portalocker")
portalocker_stub.LOCK_EX = 1
portalocker_stub.lock = lambda *_a, **_k: None
portalocker_stub.unlock = lambda *_a, **_k: None
sys.modules.setdefault("portalocker", portalocker_stub)

bs4_stub = types.ModuleType("bs4")
bs4_stub.BeautifulSoup = type("BeautifulSoup", (), {})
sys.modules.setdefault("bs4", bs4_stub)

class _Flask:
    def __init__(self, *_a, **_k):
        pass

    def route(self, *args, **kwargs):  # noqa: D401
        """Return a no-op decorator."""

        def decorator(fn):
            return fn

        return decorator

    def run(self, *args, **kwargs):  # noqa: D401
        """No-op run."""

        return None


flask_stub = types.ModuleType("flask")
flask_stub.Flask = _Flask
flask_stub.jsonify = lambda obj=None, *a, **k: obj
sys.modules.setdefault("flask", flask_stub)

cachetools_stub = types.ModuleType("cachetools")


class _TTLCache(dict):
    def __init__(self, maxsize=128, ttl=600):
        super().__init__()
        self.maxsize = maxsize
        self.ttl = ttl

    def __setitem__(self, key, value):  # noqa: D401
        """Insert while enforcing a crude max size."""

        if len(self) >= self.maxsize:
            try:
                first_key = next(iter(self))
                super().__delitem__(first_key)
            except StopIteration:  # pragma: no cover - defensive
                pass
        super().__setitem__(key, value)


cachetools_stub.TTLCache = _TTLCache
sys.modules.setdefault("cachetools", cachetools_stub)

import ai_trading.core.bot_engine as eng


def test_run_all_trades_no_warning_with_valid_api(monkeypatch):
    """A valid client with get_orders should not trigger a warning."""

    # Stub Alpaca modules so the shim translates status -> GetOrdersRequest
    enums_mod = types.ModuleType("alpaca.trading.enums")
    requests_mod = types.ModuleType("alpaca.trading.requests")

    class OrderStatus:
        OPEN = "open"

    class GetOrdersRequest:
        def __init__(self, *, statuses=None):
            self.statuses = statuses

    enums_mod.OrderStatus = OrderStatus
    requests_mod.GetOrdersRequest = GetOrdersRequest
    monkeypatch.setitem(sys.modules, "alpaca", types.ModuleType("alpaca"))
    monkeypatch.setitem(sys.modules, "alpaca.trading", types.ModuleType("alpaca.trading"))
    monkeypatch.setitem(sys.modules, "alpaca.trading.enums", enums_mod)
    monkeypatch.setitem(sys.modules, "alpaca.trading.requests", requests_mod)

    class DummyAPI:
        def __init__(self):
            self.called_with: dict | None = None

        def get_orders(self, *args, **kwargs):  # noqa: D401
            """Capture forwarded kwargs."""
            self.called_with = kwargs
            return []

        def cancel_order(self, *args, **kwargs):  # noqa: D401 - stub
            """Provide minimal cancel capability for validation."""
            return None

    class DummyRiskEngine:
        def wait_for_exposure_update(self, timeout: float) -> None:  # noqa: D401
            """No-op risk update."""

    state = eng.BotState()
    api = DummyAPI()
    runtime = types.SimpleNamespace(api=api, risk_engine=DummyRiskEngine())

    # Minimal patches to isolate the order-check logic
    monkeypatch.setattr(eng, "_ensure_alpaca_classes", lambda: None)
    monkeypatch.setattr(eng, "_init_metrics", lambda: None)
    monkeypatch.setattr(eng, "is_market_open", lambda: True)
    monkeypatch.setattr(eng, "ensure_alpaca_attached", lambda runtime: None)
    monkeypatch.setattr(eng, "check_pdt_rule", lambda runtime: False)
    monkeypatch.setattr(eng, "get_strategies", lambda: [])
    monkeypatch.setattr(eng, "get_verbose_logging", lambda: False)

    def _raise_df(*_a, **_k):
        raise eng.DataFetchError("boom")

    monkeypatch.setattr(eng, "_prepare_run", _raise_df)
    monkeypatch.setattr(eng.CFG, "log_market_fetch", False, raising=False)
    stub_fetcher = types.SimpleNamespace()
    monkeypatch.setattr(
        eng.data_fetcher_module, "build_fetcher", lambda *_a, **_k: stub_fetcher
    )
    monkeypatch.setattr(eng, "data_fetcher", stub_fetcher, raising=False)

    class DummyLock:
        def acquire(self, blocking: bool = False) -> bool:  # noqa: D401
            """Always acquire."""

            return True

        def release(self) -> None:
            """No-op release."""

    monkeypatch.setattr(eng, "run_lock", DummyLock())

    warn_mock = Mock()
    info_mock = Mock()
    monkeypatch.setattr(eng.logger_once, "warning", warn_mock)
    monkeypatch.setattr(eng.logger_once, "info", info_mock)

    eng.run_all_trades_worker(state, runtime)

    info_mock.assert_called_once_with("API_GET_ORDERS_MAPPED", key="alpaca_get_orders_mapped")
    warn_mock.assert_called_once_with("ALPACA_API_ADAPTER", key="alpaca_api_adapter")
    assert api.called_with is not None
    assert "filter" in api.called_with
    assert api.called_with["filter"].statuses == [OrderStatus.OPEN]


def test_run_all_trades_creates_trade_log(tmp_path, monkeypatch):
    """Launching the bot should create the trade log with CSV headers."""

    enums_mod = types.ModuleType("alpaca.trading.enums")
    requests_mod = types.ModuleType("alpaca.trading.requests")

    class OrderStatus:
        OPEN = "open"

    class GetOrdersRequest:
        def __init__(self, *, statuses=None):
            self.statuses = statuses

    enums_mod.OrderStatus = OrderStatus
    requests_mod.GetOrdersRequest = GetOrdersRequest
    monkeypatch.setitem(sys.modules, "alpaca", types.ModuleType("alpaca"))
    monkeypatch.setitem(sys.modules, "alpaca.trading", types.ModuleType("alpaca.trading"))
    monkeypatch.setitem(sys.modules, "alpaca.trading.enums", enums_mod)
    monkeypatch.setitem(sys.modules, "alpaca.trading.requests", requests_mod)

    class DummyAPI:
        def get_orders(self, *args, **kwargs):
            return []

        def cancel_order(self, *args, **kwargs):  # noqa: D401 - stub
            """Provide minimal cancel capability for validation."""
            return None

    class DummyRiskEngine:
        def wait_for_exposure_update(self, timeout: float) -> None:
            pass

    state = eng.BotState()
    runtime = types.SimpleNamespace(api=DummyAPI(), risk_engine=DummyRiskEngine())

    monkeypatch.setattr(eng, "_ensure_alpaca_classes", lambda: None)
    monkeypatch.setattr(eng, "_init_metrics", lambda: None)
    monkeypatch.setattr(eng, "is_market_open", lambda: True)
    monkeypatch.setattr(eng, "ensure_alpaca_attached", lambda _rt: None)
    monkeypatch.setattr(eng, "check_pdt_rule", lambda _rt: False)
    monkeypatch.setattr(eng, "get_strategies", lambda: [])
    monkeypatch.setattr(eng, "get_verbose_logging", lambda: False)
    monkeypatch.setattr(eng.CFG, "log_market_fetch", False, raising=False)
    stub_fetcher = types.SimpleNamespace()
    monkeypatch.setattr(
        eng.data_fetcher_module, "build_fetcher", lambda *_a, **_k: stub_fetcher
    )
    monkeypatch.setattr(eng, "data_fetcher", stub_fetcher, raising=False)

    class DummyLock:
        def acquire(self, blocking: bool = False) -> bool:
            return True

        def release(self) -> None:
            pass

    monkeypatch.setattr(eng, "run_lock", DummyLock())

    def _raise_df(*_a, **_k):
        raise eng.DataFetchError("boom")

    monkeypatch.setattr(eng, "_prepare_run", _raise_df)

    trade_log = tmp_path / "trades.csv"
    reward_log = tmp_path / "reward.csv"
    monkeypatch.setattr(eng, "TRADE_LOG_FILE", str(trade_log))
    monkeypatch.setattr(eng, "REWARD_LOG_FILE", str(reward_log))
    monkeypatch.setattr(eng, "_TRADE_LOGGER_SINGLETON", None)
    monkeypatch.setattr(eng, "_global_ctx", None)

    eng.run_all_trades_worker(state, runtime)

    assert trade_log.exists()
    assert (
        trade_log.read_text().splitlines()[0]
        == "symbol,entry_time,entry_price,exit_time,exit_price,qty,side,strategy,classification,signal_tags,confidence,reward"
    )


def test_run_multi_strategy_forwards_price_to_execution(monkeypatch):
    """Ensure sizing price is forwarded to execute_order."""

    expected_price = 123.45

    class DummyExecutionEngine:
        def __init__(self):
            self.calls: list[dict] = []
            self.end_cycle_called = 0

        def execute_order(
            self,
            symbol,
            side,
            qty,
            *,
            price=None,
            asset_class=None,
            **kwargs,
        ):  # noqa: D401
            """Capture the order payload."""

            self.calls.append(
                {
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "price": price,
                    "asset_class": asset_class,
                    "extra": kwargs,
                }
            )

        def end_cycle(self) -> None:
            self.end_cycle_called += 1

    class DummyRiskEngine:
        def __init__(self):
            self.last_price: float | None = None
            self.registered: list = []

        def position_exists(self, api, symbol):  # noqa: D401
            """Always allow new positions."""

            return False

        def position_size(self, sig, cash, price):  # noqa: D401
            """Record the price used for sizing and return a fixed qty."""

            self.last_price = price
            return 7

        def register_fill(self, sig):  # noqa: D401
            """Track registered fills."""

            self.registered.append(sig)

    class DummyStrategy:
        name = "stub"

        def generate(self, ctx):  # noqa: D401
            """Return a single long signal."""

            return [
                types.SimpleNamespace(
                    symbol="AAPL",
                    side="buy",
                    confidence=0.9,
                    strategy="stub",
                    weight=1.0,
                    asset_class="equity",
                )
            ]

    class DummyAllocator:
        def allocate(self, signals_by_strategy):  # noqa: D401
            """Flatten strategy outputs."""

            return [
                sig for signals in signals_by_strategy.values() for sig in signals
            ]

    class DummyAPI:
        def list_positions(self):  # noqa: D401
            """Return an empty portfolio."""

            return []

        def get_account(self):  # noqa: D401
            """Return a cash-only account."""

            return types.SimpleNamespace(cash=10_000)

        def cancel_order(self, *args, **kwargs):  # noqa: D401 - stub
            """Provide minimal cancel capability for validation."""
            return None

    class DummyDataClient:
        def __init__(self):
            self.requests: list = []

        def get_stock_latest_quote(self, req):  # noqa: D401
            """Record the request and return a static quote."""

            self.requests.append(req)
            return types.SimpleNamespace(
                ask_price=expected_price, bid_price=expected_price
            )

    dummy_exec = DummyExecutionEngine()
    dummy_risk = DummyRiskEngine()
    dummy_api = DummyAPI()
    dummy_data_client = DummyDataClient()

    ctx = types.SimpleNamespace(
        strategies=[DummyStrategy()],
        allocator=DummyAllocator(),
        api=dummy_api,
        data_client=dummy_data_client,
        execution_engine=dummy_exec,
        risk_engine=dummy_risk,
        data_fetcher=object(),
    )

    class DummyQuoteRequest:
        def __init__(self, symbol_or_symbols):
            self.symbol_or_symbols = symbol_or_symbols

    signals_mod = types.ModuleType("ai_trading.signals")
    signals_mod.enhance_signals_with_position_logic = (
        lambda signals, _ctx, _hold: signals
    )
    signals_mod.generate_position_hold_signals = lambda _ctx, _pos: []

    monkeypatch.setitem(sys.modules, "ai_trading.signals", signals_mod)
    monkeypatch.setattr(eng, "StockLatestQuoteRequest", DummyQuoteRequest)
    monkeypatch.setattr(eng, "to_trade_signal", lambda sig: sig)
    fetch_minute_mock = Mock(return_value=None)
    monkeypatch.setattr(eng, "fetch_minute_df_safe", fetch_minute_mock)

    eng.run_multi_strategy(ctx)

    assert dummy_exec.calls, "execute_order should be invoked"
    call = dummy_exec.calls[0]
    assert call["price"] == pytest.approx(expected_price)
    assert call["asset_class"] == "equity"
    assert dummy_risk.last_price == pytest.approx(expected_price)
