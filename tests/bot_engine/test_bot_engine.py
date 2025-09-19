import sys
import types
from datetime import UTC, datetime, time, timedelta

import pytest

if "numpy" not in sys.modules:  # pragma: no cover - optional dependency shim
    numpy_stub = types.ModuleType("numpy")
    numpy_stub.nan = float("nan")
    numpy_stub.NaN = numpy_stub.nan
    numpy_stub.random = types.SimpleNamespace(seed=lambda *_args, **_kwargs: None)
    sys.modules["numpy"] = numpy_stub

if "portalocker" not in sys.modules:  # pragma: no cover - optional dependency shim
    sys.modules["portalocker"] = types.ModuleType("portalocker")

if "bs4" not in sys.modules:  # pragma: no cover - optional dependency shim
    bs4_stub = types.ModuleType("bs4")

    class _BeautifulSoup:  # pragma: no cover - minimal placeholder
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    bs4_stub.BeautifulSoup = _BeautifulSoup
    sys.modules["bs4"] = bs4_stub

if "flask" not in sys.modules:  # pragma: no cover - optional dependency shim
    flask_stub = types.ModuleType("flask")

    class _Flask:  # pragma: no cover - minimal placeholder
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

        def route(self, *_a, **_k):  # noqa: ANN001 - placeholder
            def decorator(func):
                return func

            return decorator

    flask_stub.Flask = _Flask
    sys.modules["flask"] = flask_stub

from ai_trading.core import bot_engine


class DummyHaltManager:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def manual_halt_trading(self, reason: str) -> None:
        self.calls.append(reason)


class DummyExecutor:
    def submit(self, fn, symbol):  # noqa: ANN001 - test helper
        return types.SimpleNamespace(result=lambda: fn(symbol))


class DummyContext:
    def __init__(self, halt_manager):
        self.halt_manager = halt_manager


class TestProcessSymbol:
    @pytest.fixture(autouse=True)
    def _patch_env(self, monkeypatch):
        monkeypatch.setattr(bot_engine, "get_env", lambda key, default=None: default, raising=False)

    def test_process_symbol_skips_on_all_nan_close(self, monkeypatch):
        state = bot_engine.BotState()
        state.position_cache = {}
        bot_engine.state = state

        dummy_halt = DummyHaltManager()
        dummy_ctx = DummyContext(dummy_halt)
        monkeypatch.setattr(bot_engine, "get_ctx", lambda: dummy_ctx)
        monkeypatch.setattr(bot_engine, "is_market_open", lambda: True)
        monkeypatch.setattr(bot_engine, "ensure_final_bar", lambda *_, **__: True)
        monkeypatch.setattr(bot_engine, "log_skip_cooldown", lambda *_, **__: None)
        monkeypatch.setattr(
            bot_engine,
            "skipped_duplicates",
            types.SimpleNamespace(inc=lambda: None),
            raising=False,
        )
        monkeypatch.setattr(
            bot_engine,
            "skipped_cooldown",
            types.SimpleNamespace(inc=lambda: None),
            raising=False,
        )
        monkeypatch.setattr(bot_engine.executors, "_ensure_executors", lambda: None)
        monkeypatch.setattr(bot_engine, "prediction_executor", DummyExecutor(), raising=False)
        monkeypatch.setattr(bot_engine, "_safe_trade", lambda *_, **__: None)

        def _raise_nan(_symbol: str):
            err = bot_engine.DataFetchError("close_column_all_nan")
            setattr(err, "fetch_reason", "close_column_all_nan")
            raise err

        monkeypatch.setattr(bot_engine, "fetch_minute_df_safe", _raise_nan)

        processed, row_counts = bot_engine._process_symbols(["AAPL"], 1000.0, None, True)

        assert processed == []
        assert row_counts == {}
        assert dummy_halt.calls == ["AAPL:empty_frame"]

    def test_process_symbols_early_exit_when_trade_quota_exhausted(
        self, monkeypatch, caplog
    ):
        state = bot_engine.BotState()
        state.position_cache = {}
        now = datetime.now(UTC)
        state.trade_history = [
            ("SYM", now - timedelta(minutes=5))
            for _ in range(bot_engine.MAX_TRADES_PER_HOUR)
        ]
        bot_engine.state = state

        dummy_halt = DummyHaltManager()
        dummy_ctx = DummyContext(dummy_halt)
        monkeypatch.setattr(bot_engine, "get_ctx", lambda: dummy_ctx)
        monkeypatch.setattr(bot_engine, "is_market_open", lambda: True)
        monkeypatch.setattr(bot_engine, "ensure_final_bar", lambda *_, **__: True)
        monkeypatch.setattr(bot_engine, "log_skip_cooldown", lambda *_, **__: None)
        monkeypatch.setattr(
            bot_engine,
            "skipped_duplicates",
            types.SimpleNamespace(inc=lambda: None),
            raising=False,
        )
        monkeypatch.setattr(
            bot_engine,
            "skipped_cooldown",
            types.SimpleNamespace(inc=lambda: None),
            raising=False,
        )
        monkeypatch.setattr(bot_engine.executors, "_ensure_executors", lambda: None)
        monkeypatch.setattr(bot_engine, "prediction_executor", DummyExecutor(), raising=False)
        monkeypatch.setattr(bot_engine, "_safe_trade", lambda *_, **__: None)

        def _fail_fetch(symbol: str):  # pragma: no cover - should not be called
            raise AssertionError(f"fetch_minute_df_safe called for {symbol}")

        monkeypatch.setattr(bot_engine, "fetch_minute_df_safe", _fail_fetch)

        caplog.set_level("INFO")

        processed, row_counts = bot_engine._process_symbols(
            ["AAPL", "MSFT"], 1000.0, None, True
        )

        assert processed == []
        assert row_counts == {}
        assert any(
            record.message == "TRADE_QUOTA_EXHAUSTED_SKIP" for record in caplog.records
        )


def test_trade_logic_uses_fallback_when_primary_disabled(monkeypatch):
    pd = pytest.importorskip("pandas")

    symbol = "AAPL"
    orders = []

    class _DummyAPI:
        def list_positions(self):  # noqa: D401, ANN001 - minimal stub
            return []

        def get_account(self):  # noqa: D401 - minimal stub
            return types.SimpleNamespace(equity=100000.0, portfolio_value=100000.0)

    class _Logger:
        def __init__(self) -> None:
            self.entries: list[tuple] = []

        def log_entry(self, *args, **kwargs):  # noqa: D401 - capture call
            self.entries.append((args, kwargs))

        def log_exit(self, *args, **kwargs):  # noqa: D401 - unused in test
            return None

    feat_df = pd.DataFrame(
        {
            "close": [100.0, 101.0],
            "open": [99.5, 100.5],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "volume": [1_000, 1_200],
            "macd": [0.1, 0.2],
            "atr": [1.0, 1.0],
            "vwap": [100.2, 100.6],
            "macds": [0.05, 0.05],
            "sma_50": [99.0, 99.5],
            "sma_200": [95.0, 95.5],
        }
    )

    ctx = types.SimpleNamespace(
        signal_manager=types.SimpleNamespace(last_components=[]),
        data_fetcher=types.SimpleNamespace(
            get_daily_df=lambda *_a, **_k: feat_df.copy()
        ),
        portfolio_weights={},
        api=_DummyAPI(),
        trade_logger=_Logger(),
        take_profit_targets={},
        stop_targets={},
        market_open=time(6, 30),
        market_close=time(13, 0),
        rebalance_buys={},
        config=types.SimpleNamespace(exposure_cap_aggressive=0.9),
    )

    state = bot_engine.BotState()
    state.position_cache = {}

    monkeypatch.setenv("PYTEST_RUNNING", "")
    monkeypatch.setenv("TESTING", "")
    monkeypatch.setenv("DRY_RUN", "")

    monkeypatch.setattr(bot_engine, "pre_trade_checks", lambda *a, **k: True)

    def _fake_fetch(*_a, **_k):
        return feat_df.copy(), feat_df.copy(), None

    def _fake_eval(ctx_obj, state_obj, _df, _symbol, _model):
        ctx_obj.signal_manager.last_components = [(1, 0.8, "fallback_test")]
        return 1.0, 0.8, "fallback_test"

    monkeypatch.setattr(bot_engine, "_fetch_feature_data", _fake_fetch)
    monkeypatch.setattr(bot_engine, "_evaluate_trade_signal", _fake_eval)
    monkeypatch.setattr(bot_engine, "_exit_positions_if_needed", lambda *a, **k: False)
    monkeypatch.setattr(bot_engine, "_check_trade_frequency_limits", lambda *a, **k: False)
    monkeypatch.setattr(bot_engine, "get_trade_cooldown_min", lambda: 0)
    monkeypatch.setattr(bot_engine, "get_buy_threshold", lambda: 0.5)
    monkeypatch.setattr(bot_engine, "_current_qty", lambda *a, **k: 0)
    monkeypatch.setattr(
        bot_engine, "_apply_sector_cap_qty", lambda _ctx, _symbol, qty, _price: qty
    )

    def _fake_scaled_atr_stop(*a, **_k):
        entry_price = a[0] if a else _k.get("entry_price", 0.0)
        return entry_price * 0.95, entry_price * 1.05

    monkeypatch.setattr(bot_engine, "scaled_atr_stop", _fake_scaled_atr_stop)
    monkeypatch.setattr(bot_engine, "is_high_vol_regime", lambda: False)
    monkeypatch.setattr(bot_engine, "get_take_profit_factor", lambda: 1.0)
    monkeypatch.setattr(bot_engine, "_record_trade_in_frequency_tracker", lambda *a, **k: None)
    monkeypatch.setattr(bot_engine, "_PRICE_SOURCE", {}, raising=False)

    def _fake_latest_price(sym, *, prefer_backup=False):
        if prefer_backup:
            bot_engine._PRICE_SOURCE[sym] = "yahoo"
            return 101.0
        bot_engine._PRICE_SOURCE[sym] = "alpaca_ask"
        return 0.0

    monkeypatch.setattr(bot_engine, "get_latest_price", _fake_latest_price)

    def _fake_submit(ctx_obj, sym, qty, side, price=None):  # noqa: D401 - capture order
        orders.append((sym, qty, side, price))
        return types.SimpleNamespace(id="order-1")

    monkeypatch.setattr(bot_engine, "submit_order", _fake_submit)
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "is_primary_provider_enabled",
        lambda: False,
        raising=False,
    )

    result = bot_engine.trade_logic(
        ctx,
        state,
        symbol,
        balance=100000.0,
        model=None,
        regime_ok=True,
    )

    assert result is True
    assert orders and orders[0][0] == symbol
    assert "alpaca" in state.degraded_providers
    assert bot_engine._PRICE_SOURCE[symbol] == "yahoo"


def test_enter_long_uses_feature_close_when_quote_invalid(monkeypatch, caplog):
    pd = pytest.importorskip("pandas")

    symbol = "AAPL"
    orders: list[tuple[str, int, str, float | None]] = []

    class _DummyAPI:
        def list_positions(self):  # noqa: D401, ANN001 - minimal stub
            return []

        def get_account(self):  # noqa: D401 - minimal stub
            return types.SimpleNamespace(equity=100000.0, portfolio_value=100000.0)

    class _Logger:
        def log_entry(self, *args, **kwargs):  # noqa: D401 - capture call
            return (args, kwargs)

        def log_exit(self, *args, **kwargs):  # noqa: D401 - unused in test
            return None

    feat_df = pd.DataFrame(
        {
            "close": [100.0, 101.0],
            "open": [99.5, 100.5],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "volume": [1_000, 1_200],
            "macd": [0.1, 0.2],
            "atr": [1.0, 1.0],
            "vwap": [100.2, 100.6],
            "macds": [0.05, 0.05],
            "sma_50": [99.0, 99.5],
            "sma_200": [95.0, 95.5],
        }
    )

    ctx = types.SimpleNamespace(
        portfolio_weights={symbol: 0.02},
        api=_DummyAPI(),
        trade_logger=_Logger(),
        take_profit_targets={},
        stop_targets={},
        market_open=time(6, 30),
        market_close=time(13, 0),
        config=types.SimpleNamespace(exposure_cap_aggressive=0.9),
    )

    state = bot_engine.BotState()

    monkeypatch.delenv("PYTEST_RUNNING", raising=False)
    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.delenv("DRY_RUN", raising=False)

    monkeypatch.setattr(bot_engine, "_apply_sector_cap_qty", lambda _ctx, _sym, qty, _price: qty)
    monkeypatch.setattr(bot_engine, "scaled_atr_stop", lambda **_k: (95.95, 106.05))
    monkeypatch.setattr(bot_engine, "is_high_vol_regime", lambda: False)
    monkeypatch.setattr(bot_engine, "get_take_profit_factor", lambda: 1.0)
    monkeypatch.setattr(bot_engine, "_record_trade_in_frequency_tracker", lambda *a, **k: None)
    monkeypatch.setattr(bot_engine, "submit_order", lambda _ctx, sym, qty, side, price=None: orders.append((sym, qty, side, price)) or types.SimpleNamespace(id="order-1"))
    monkeypatch.setattr(
        bot_engine,
        "get_latest_price",
        lambda _symbol, *, prefer_backup=False: 0.0,
    )
    monkeypatch.setattr(bot_engine, "_PRICE_SOURCE", {symbol: "alpaca_ask"}, raising=False)

    caplog.set_level("WARNING")

    result = bot_engine._enter_long(
        ctx,
        state,
        symbol,
        balance=100000.0,
        feat_df=feat_df,
        final_score=1.0,
        conf=0.8,
        strat="invalid_quote_test",
    )

    assert result is True
    assert orders and orders[0][3] == pytest.approx(feat_df["close"].iloc[-1])
    assert any("FALLBACK_TO_FEATURE_CLOSE" in rec.message for rec in caplog.records)
    assert not any("computed qty <= 0" in rec.message for rec in caplog.records)
    assert bot_engine._PRICE_SOURCE[symbol] == "feature_close"


def _build_dummy_long_context(pd, symbol):
    class _DummyAPI:
        def list_positions(self):
            return []

        def get_account(self):
            return types.SimpleNamespace(equity=100000.0, portfolio_value=100000.0)

    class _Logger:
        def log_entry(self, *args, **kwargs):
            return (args, kwargs)

        def log_exit(self, *args, **kwargs):
            return None

    feat_df = pd.DataFrame(
        {
            "close": [100.0, 101.0],
            "open": [99.5, 100.5],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "volume": [1_000, 1_200],
            "macd": [0.1, 0.2],
            "atr": [1.0, 1.0],
            "vwap": [100.2, 100.6],
            "macds": [0.05, 0.05],
            "sma_50": [99.0, 99.5],
            "sma_200": [95.0, 95.5],
        }
    )

    ctx = types.SimpleNamespace(
        portfolio_weights={symbol: 0.02},
        api=_DummyAPI(),
        trade_logger=_Logger(),
        take_profit_targets={},
        stop_targets={},
        market_open=time(6, 30),
        market_close=time(13, 0),
        config=types.SimpleNamespace(exposure_cap_aggressive=0.9),
    )

    state = bot_engine.BotState()

    return ctx, state, feat_df


def _build_dummy_short_context(pd, symbol):
    class _DummyAPI:
        def get_asset(self, sym):
            assert sym == symbol
            return types.SimpleNamespace(shortable=True, shortable_shares=50)

    class _Logger:
        def log_entry(self, *args, **kwargs):
            return (args, kwargs)

        def log_exit(self, *args, **kwargs):
            return None

    feat_df = _build_dummy_long_context(pd, symbol)[2]

    ctx = types.SimpleNamespace(
        api=_DummyAPI(),
        trade_logger=_Logger(),
        take_profit_targets={},
        stop_targets={},
        market_open=time(6, 30),
        market_close=time(13, 0),
    )

    state = bot_engine.BotState()

    return ctx, state, feat_df


def test_enter_long_skips_fallback_log_for_alpaca_sources(monkeypatch, caplog):
    pd = pytest.importorskip("pandas")

    symbol = "AAPL"
    orders: list[tuple[str, int, str, float | None]] = []
    ctx, state, feat_df = _build_dummy_long_context(pd, symbol)

    monkeypatch.delenv("PYTEST_RUNNING", raising=False)
    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.delenv("DRY_RUN", raising=False)

    monkeypatch.setattr(
        bot_engine,
        "_apply_sector_cap_qty",
        lambda _ctx, _sym, qty, _price: qty,
    )
    monkeypatch.setattr(bot_engine, "scaled_atr_stop", lambda **_k: (95.95, 106.05))
    monkeypatch.setattr(bot_engine, "is_high_vol_regime", lambda: False)
    monkeypatch.setattr(bot_engine, "get_take_profit_factor", lambda: 1.0)
    monkeypatch.setattr(
        bot_engine,
        "_record_trade_in_frequency_tracker",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        bot_engine,
        "submit_order",
        lambda _ctx, sym, qty, side, price=None: orders.append((sym, qty, side, price))
        or types.SimpleNamespace(id="order-1"),
    )
    monkeypatch.setattr(
        bot_engine,
        "_resolve_order_quote",
        lambda *_a, **_k: (101.0, "alpaca_snapshot_primary"),
    )

    caplog.set_level("INFO")

    result = bot_engine._enter_long(
        ctx,
        state,
        symbol,
        balance=100000.0,
        feat_df=feat_df,
        final_score=1.0,
        conf=0.8,
        strat="alpaca_source_test",
    )

    assert result is True
    assert orders and orders[0][3] == pytest.approx(101.0)
    assert not any(
        record.message == "ORDER_USING_FALLBACK_PRICE" for record in caplog.records
    )


def test_enter_long_logs_fallback_for_non_alpaca_sources(monkeypatch, caplog):
    pd = pytest.importorskip("pandas")

    symbol = "AAPL"
    orders: list[tuple[str, int, str, float | None]] = []
    ctx, state, feat_df = _build_dummy_long_context(pd, symbol)

    monkeypatch.delenv("PYTEST_RUNNING", raising=False)
    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.delenv("DRY_RUN", raising=False)

    monkeypatch.setattr(
        bot_engine,
        "_apply_sector_cap_qty",
        lambda _ctx, _sym, qty, _price: qty,
    )
    monkeypatch.setattr(bot_engine, "scaled_atr_stop", lambda **_k: (95.95, 106.05))
    monkeypatch.setattr(bot_engine, "is_high_vol_regime", lambda: False)
    monkeypatch.setattr(bot_engine, "get_take_profit_factor", lambda: 1.0)
    monkeypatch.setattr(
        bot_engine,
        "_record_trade_in_frequency_tracker",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        bot_engine,
        "submit_order",
        lambda _ctx, sym, qty, side, price=None: orders.append((sym, qty, side, price))
        or types.SimpleNamespace(id="order-1"),
    )
    monkeypatch.setattr(
        bot_engine,
        "_resolve_order_quote",
        lambda *_a, **_k: (101.0, "iex"),
    )

    caplog.set_level("INFO")

    result = bot_engine._enter_long(
        ctx,
        state,
        symbol,
        balance=100000.0,
        feat_df=feat_df,
        final_score=1.0,
        conf=0.8,
        strat="fallback_source_test",
    )

    assert result is True
    assert orders and orders[0][3] == pytest.approx(101.0)
    assert any(
        record.message == "ORDER_USING_FALLBACK_PRICE" for record in caplog.records
    )


def test_enter_short_skips_fallback_log_for_alpaca_sources(monkeypatch, caplog):
    pd = pytest.importorskip("pandas")

    symbol = "AAPL"
    orders: list[tuple[str, int, str, float | None]] = []
    ctx, state, feat_df = _build_dummy_short_context(pd, symbol)

    monkeypatch.delenv("PYTEST_RUNNING", raising=False)
    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.delenv("DRY_RUN", raising=False)

    monkeypatch.setattr(bot_engine, "calculate_entry_size", lambda *a, **k: 5)
    monkeypatch.setattr(
        bot_engine,
        "_apply_sector_cap_qty",
        lambda _ctx, _sym, qty, _price: qty,
    )
    monkeypatch.setattr(
        bot_engine,
        "submit_order",
        lambda _ctx, sym, qty, side, price=None: orders.append((sym, qty, side, price))
        or types.SimpleNamespace(id="order-1"),
    )
    monkeypatch.setattr(
        bot_engine,
        "_resolve_order_quote",
        lambda *_a, **_k: (101.0, "alpaca_deep_book"),
    )

    caplog.set_level("INFO")

    result = bot_engine._enter_short(
        ctx,
        state,
        symbol,
        feat_df=feat_df,
        final_score=-1.0,
        conf=-0.8,
        strat="alpaca_short_source_test",
    )

    assert result is True
    assert orders and orders[0][3] == pytest.approx(101.0)
    assert not any(
        record.message == "ORDER_USING_FALLBACK_PRICE" for record in caplog.records
    )


def test_enter_short_logs_fallback_for_non_alpaca_sources(monkeypatch, caplog):
    pd = pytest.importorskip("pandas")

    symbol = "AAPL"
    orders: list[tuple[str, int, str, float | None]] = []
    ctx, state, feat_df = _build_dummy_short_context(pd, symbol)

    monkeypatch.delenv("PYTEST_RUNNING", raising=False)
    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.delenv("DRY_RUN", raising=False)

    monkeypatch.setattr(bot_engine, "calculate_entry_size", lambda *a, **k: 5)
    monkeypatch.setattr(
        bot_engine,
        "_apply_sector_cap_qty",
        lambda _ctx, _sym, qty, _price: qty,
    )
    monkeypatch.setattr(
        bot_engine,
        "submit_order",
        lambda _ctx, sym, qty, side, price=None: orders.append((sym, qty, side, price))
        or types.SimpleNamespace(id="order-1"),
    )
    monkeypatch.setattr(
        bot_engine,
        "_resolve_order_quote",
        lambda *_a, **_k: (101.0, "polygon"),
    )

    caplog.set_level("INFO")

    result = bot_engine._enter_short(
        ctx,
        state,
        symbol,
        feat_df=feat_df,
        final_score=-1.0,
        conf=-0.8,
        strat="fallback_short_source_test",
    )

    assert result is True
    assert orders and orders[0][3] == pytest.approx(101.0)
    assert any(
        record.message == "ORDER_USING_FALLBACK_PRICE" for record in caplog.records
    )
