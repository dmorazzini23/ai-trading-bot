import sys
import types
from datetime import UTC, datetime, time, timedelta
from pathlib import Path

from unittest.mock import MagicMock

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

from ai_trading.config.runtime import reload_trading_config
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


def test_get_trade_logger_uses_fallback_when_primary_unwritable(monkeypatch, tmp_path):
    primary_dir = tmp_path / "primary"
    fallback_dir = tmp_path / "fallback"
    primary_dir.mkdir()
    fallback_dir.mkdir()
    primary_file = primary_dir / "trades.csv"
    fallback_path = fallback_dir / "trades.csv"

    constructed_paths: list[str] = []
    primary_dir_resolved = primary_dir.resolve()

    def fake_is_dir_writable(path: str) -> bool:
        try:
            resolved = Path(path).resolve()
        except OSError:
            resolved = Path(path)
        if resolved == primary_dir_resolved:
            return False
        return True

    class _StubTradeLogger:
        def __init__(self, path, *args, **kwargs):
            self.path = str(path)
            constructed_paths.append(self.path)

    monkeypatch.setattr(bot_engine, "_is_dir_writable", fake_is_dir_writable)
    monkeypatch.setattr(bot_engine, "TradeLogger", _StubTradeLogger)
    monkeypatch.setattr(
        bot_engine,
        "_compute_user_state_trade_log_path",
        lambda filename="trades.csv": str(fallback_path),
    )
    monkeypatch.setattr(bot_engine, "_TRADE_LOGGER_SINGLETON", None, raising=False)
    monkeypatch.setattr(bot_engine, "_TRADE_LOG_FALLBACK_PATH", None, raising=False)
    monkeypatch.setattr(bot_engine, "TRADE_LOG_FILE", str(primary_file), raising=False)

    logger = bot_engine.get_trade_logger()

    assert constructed_paths[0] == str(primary_file)
    assert constructed_paths[-1] == str(fallback_path)
    assert logger.path == str(fallback_path)
    assert bot_engine.TRADE_LOG_FILE == str(fallback_path)
    assert bot_engine._TRADE_LOG_FALLBACK_PATH == str(fallback_path)


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
        bot_engine._PRICE_SOURCE[sym] = bot_engine._ALPACA_DISABLED_SENTINEL
        return None

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


def test_get_latest_price_skips_primary_during_cooldown(monkeypatch):
    monkeypatch.setattr(bot_engine, "_PRICE_SOURCE", {}, raising=False)
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "is_primary_provider_enabled",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(bot_engine, "is_alpaca_service_available", lambda: True)

    def _fail_alpaca_symbols():  # pragma: no cover - should not be invoked
        raise AssertionError("Alpaca should not be queried during cooldown")

    monkeypatch.setattr(bot_engine, "_alpaca_symbols", _fail_alpaca_symbols)

    price = bot_engine.get_latest_price("AAPL")

    assert price is None
    assert bot_engine._PRICE_SOURCE["AAPL"] == bot_engine._ALPACA_DISABLED_SENTINEL


def test_get_latest_price_prefers_backup_when_primary_disabled(monkeypatch):
    monkeypatch.setattr(bot_engine, "_PRICE_SOURCE", {}, raising=False)
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "is_primary_provider_enabled",
        lambda: False,
        raising=False,
    )

    def _fail_alpaca_symbols():  # pragma: no cover - should not be invoked
        raise AssertionError("Alpaca should not be queried when disabled")

    monkeypatch.setattr(bot_engine, "_alpaca_symbols", _fail_alpaca_symbols)

    called: dict[str, int] = {"backup": 0}

    def _fake_backup_get_bars(symbol, start, end, interval):  # noqa: D401, ANN001
        called["backup"] += 1
        return object()

    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "_backup_get_bars",
        _fake_backup_get_bars,
    )
    monkeypatch.setattr(bot_engine, "get_latest_close", lambda _df: 123.45)

    price = bot_engine.get_latest_price("AAPL", prefer_backup=True)

    assert price == 123.45
    assert called["backup"] == 1
    assert bot_engine._PRICE_SOURCE["AAPL"] == "yahoo"


def test_enter_long_skips_when_primary_disabled(monkeypatch, caplog):
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
    def _fake_latest_price(_symbol, *, prefer_backup=False):
        bot_engine._PRICE_SOURCE[_symbol] = bot_engine._ALPACA_DISABLED_SENTINEL
        return None

    monkeypatch.setattr(bot_engine, "get_latest_price", _fake_latest_price)
    monkeypatch.setattr(bot_engine, "_PRICE_SOURCE", {symbol: bot_engine._ALPACA_DISABLED_SENTINEL}, raising=False)
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "is_primary_provider_enabled",
        lambda: False,
        raising=False,
    )

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
    assert orders == []
    assert any(
        record.message == "SKIP_ORDER_ALPACA_UNAVAILABLE" for record in caplog.records
    )


    assert not any(
        record.message == "FALLBACK_TO_FEATURE_CLOSE" for record in caplog.records
    )


class TestBuyingPowerEnforcement:
    def test_enforce_buying_power_allows_full_quantity(self):
        account = types.SimpleNamespace(buying_power="10000")
        ctx = types.SimpleNamespace(api=types.SimpleNamespace(get_account=lambda: account))

        qty, available = bot_engine._enforce_buying_power_limit(ctx, account, "buy", 100.0, 50)

        assert qty == 50
        assert available == pytest.approx(10000.0)

    def test_enforce_buying_power_downsizes_quantity(self):
        account = types.SimpleNamespace(buying_power=900)
        ctx = types.SimpleNamespace(api=types.SimpleNamespace(get_account=lambda: account))

        qty, available = bot_engine._enforce_buying_power_limit(ctx, account, "buy", 100.0, 15)

        assert qty == 9  # floor division of available // price
        assert available == pytest.approx(900.0)

    def test_enforce_buying_power_keeps_quantity_when_unavailable(self):
        account = types.SimpleNamespace(shorting_power=0)
        ctx = types.SimpleNamespace(api=types.SimpleNamespace(get_account=lambda: account))

        qty, available = bot_engine._enforce_buying_power_limit(ctx, account, "sell_short", 50.0, 10)

        assert qty == 10
        assert available == 0.0


def test_enter_long_skips_when_alpaca_auth_failed(monkeypatch, caplog):
    pd = pytest.importorskip("pandas")

    symbol = "AAPL"
    orders: list[tuple[str, int, str, float | None]] = []
    ctx, state, feat_df = _build_dummy_long_context(pd, symbol)

    monkeypatch.delenv("PYTEST_RUNNING", raising=False)
    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.delenv("DRY_RUN", raising=False)

    monkeypatch.setattr(
        bot_engine,
        "submit_order",
        lambda _ctx, sym, qty, side, price=None: orders.append(
            (sym, qty, side, price)
        )
        or types.SimpleNamespace(id="order-1"),
    )
    monkeypatch.setattr(
        bot_engine,
        "_resolve_order_quote",
        lambda *_a, **_k: (None, "alpaca_auth_failed"),
    )

    caplog.set_level("WARNING")

    result = bot_engine._enter_long(
        ctx,
        state,
        symbol,
        balance=100000.0,
        feat_df=feat_df,
        final_score=1.0,
        conf=0.8,
        strat="auth_failed_test",
    )

    assert result is True
    assert orders == []
    assert symbol in state.auth_skipped_symbols
    assert any(
        record.message == "SKIP_ORDER_ALPACA_UNAVAILABLE" for record in caplog.records
    )


def test_enter_long_skips_when_backup_quote_degraded(monkeypatch, caplog):
    pd = pytest.importorskip("pandas")

    symbol = "AAPL"
    orders: list[tuple[str, int, str, float | None]] = []
    ctx, state, feat_df = _build_dummy_long_context(pd, symbol)

    monkeypatch.delenv("PYTEST_RUNNING", raising=False)
    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.delenv("DRY_RUN", raising=False)

    monkeypatch.setattr(
        bot_engine,
        "submit_order",
        lambda _ctx, sym, qty, side, price=None: orders.append((sym, qty, side, price))
        or types.SimpleNamespace(id="order-1"),
    )
    monkeypatch.setattr(
        bot_engine,
        "_resolve_order_quote",
        lambda *_a, **_k: bot_engine._PRICE_SOURCE.__setitem__(
            symbol, "yahoo_invalid"
        )
        or (None, "yahoo_invalid"),
    )
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "is_primary_provider_enabled",
        lambda: False,
        raising=False,
    )

    caplog.set_level("WARNING")

    result = bot_engine._enter_long(
        ctx,
        state,
        symbol,
        balance=100000.0,
        feat_df=feat_df,
        final_score=1.0,
        conf=0.8,
        strat="degraded_quote_test",
    )

    assert result is True
    assert orders == []
    assert "alpaca" in state.degraded_providers
    assert symbol in state.degraded_providers
    assert any(
        record.message == "SKIP_ORDER_DEGRADED_QUOTE" for record in caplog.records
    )
    assert not any(
        record.message == "FALLBACK_TO_FEATURE_CLOSE" for record in caplog.records
    )


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


def _patch_price_gate_common(monkeypatch, symbol: str, price: float) -> None:
    monkeypatch.delenv("PYTEST_RUNNING", raising=False)
    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.delenv("DRY_RUN", raising=False)
    monkeypatch.setattr(
        bot_engine,
        "_resolve_order_quote",
        lambda _symbol, prefer_backup=False: (price, "alpaca_ask"),
    )
    monkeypatch.setattr(bot_engine, "_stock_quote_request_ready", lambda: True)
    monkeypatch.setattr(
        bot_engine,
        "_evaluate_data_gating",
        lambda *a, **k: types.SimpleNamespace(
            block=False, reasons=tuple(), annotations={}, size_cap=None
        ),
    )
    monkeypatch.setattr(bot_engine, "_set_price_source", lambda *_a, **_k: None)
    monkeypatch.setattr(bot_engine, "_clear_cached_yahoo_fallback", lambda *_a, **_k: None)


def test_enter_long_price_gate_missing_bid_ask(monkeypatch, caplog):
    pd = pytest.importorskip("pandas")

    symbol = "AAPL"
    ctx, state, feat_df = _build_dummy_long_context(pd, symbol)
    ctx.data_client = object()

    monkeypatch.setenv("EXECUTION_ALLOW_FALLBACK_PRICE", "0")
    reload_trading_config()

    _patch_price_gate_common(monkeypatch, symbol, 100.0)
    monkeypatch.setattr(bot_engine, "_fetch_quote", lambda *_a, **_k: types.SimpleNamespace())
    submit_mock = MagicMock(return_value=None)
    monkeypatch.setattr(bot_engine, "submit_order", submit_mock)

    caplog.set_level("INFO")
    result = bot_engine._enter_long(
        ctx,
        state,
        symbol,
        balance=100000.0,
        feat_df=feat_df,
        final_score=1.0,
        conf=0.9,
        strat="gate_missing",
    )

    assert result is True
    assert any(
        record.message
        == "ORDER_SKIPPED_PRICE_GATED | symbol=AAPL reason=missing_bid_ask"
        for record in caplog.records
    )
    assert not any("SIGNAL_BUY" in record.message for record in caplog.records)
    submit_mock.assert_not_called()

    monkeypatch.delenv("EXECUTION_ALLOW_FALLBACK_PRICE", raising=False)
    reload_trading_config()


def test_enter_long_price_gate_stale_quote(monkeypatch, caplog):
    pd = pytest.importorskip("pandas")

    symbol = "AAPL"
    ctx, state, feat_df = _build_dummy_long_context(pd, symbol)
    ctx.data_client = object()

    monkeypatch.setenv("EXECUTION_ALLOW_FALLBACK_PRICE", "0")
    reload_trading_config()

    _patch_price_gate_common(monkeypatch, symbol, 100.0)
    stale_ts = datetime.now(UTC) - timedelta(minutes=10)
    monkeypatch.setattr(
        bot_engine,
        "_fetch_quote",
        lambda *_a, **_k: types.SimpleNamespace(
            bid_price=100.0,
            ask_price=100.5,
            timestamp=stale_ts,
        ),
    )
    submit_mock = MagicMock(return_value=None)
    monkeypatch.setattr(bot_engine, "submit_order", submit_mock)

    caplog.set_level("INFO")
    result = bot_engine._enter_long(
        ctx,
        state,
        symbol,
        balance=100000.0,
        feat_df=feat_df,
        final_score=1.0,
        conf=0.9,
        strat="gate_stale",
    )

    assert result is True
    assert any(
        record.message
        == "ORDER_SKIPPED_PRICE_GATED | symbol=AAPL reason=stale_quote"
        for record in caplog.records
    )
    assert not any("SIGNAL_BUY" in record.message for record in caplog.records)
    submit_mock.assert_not_called()

    monkeypatch.delenv("EXECUTION_ALLOW_FALLBACK_PRICE", raising=False)
    reload_trading_config()


def test_enter_long_price_gate_negative_spread(monkeypatch, caplog):
    pd = pytest.importorskip("pandas")

    symbol = "AAPL"
    ctx, state, feat_df = _build_dummy_long_context(pd, symbol)
    ctx.data_client = object()

    monkeypatch.setenv("EXECUTION_ALLOW_FALLBACK_PRICE", "0")
    reload_trading_config()

    _patch_price_gate_common(monkeypatch, symbol, 100.0)
    monkeypatch.setattr(
        bot_engine,
        "_fetch_quote",
        lambda *_a, **_k: types.SimpleNamespace(
            bid_price=101.0,
            ask_price=100.5,
            timestamp=datetime.now(UTC),
        ),
    )
    submit_mock = MagicMock(return_value=None)
    monkeypatch.setattr(bot_engine, "submit_order", submit_mock)

    caplog.set_level("INFO")
    result = bot_engine._enter_long(
        ctx,
        state,
        symbol,
        balance=100000.0,
        feat_df=feat_df,
        final_score=1.0,
        conf=0.9,
        strat="gate_negative_spread",
    )

    assert result is True
    assert any(
        record.message
        == "ORDER_SKIPPED_PRICE_GATED | symbol=AAPL reason=negative_spread"
        for record in caplog.records
    )
    assert not any("SIGNAL_BUY" in record.message for record in caplog.records)
    submit_mock.assert_not_called()

    monkeypatch.delenv("EXECUTION_ALLOW_FALLBACK_PRICE", raising=False)
    reload_trading_config()


def test_quote_gate_helper_reasons():
    now = datetime.now(UTC)
    missing = bot_engine._evaluate_quote_gate(
        types.SimpleNamespace(),
        require_bid_ask=True,
        max_age_sec=10.0,
    )
    assert not missing
    assert missing.reason == "missing_bid_ask"

    negative = bot_engine._evaluate_quote_gate(
        types.SimpleNamespace(bid_price=101.0, ask_price=100.0, timestamp=now),
        require_bid_ask=True,
        max_age_sec=10.0,
    )
    assert not negative
    assert negative.reason == "negative_spread"

    stale_ts = now - timedelta(minutes=5)
    stale = bot_engine._evaluate_quote_gate(
        types.SimpleNamespace(bid_price=100.0, ask_price=100.5, timestamp=stale_ts),
        require_bid_ask=True,
        max_age_sec=30.0,
    )
    assert not stale
    assert stale.reason == "stale_quote"

    fresh = bot_engine._evaluate_quote_gate(
        types.SimpleNamespace(bid_price=100.0, ask_price=100.2, timestamp=now),
        require_bid_ask=True,
        max_age_sec=30.0,
    )
    assert fresh
    assert fresh.reason is None


def test_enter_long_price_gate_gap_ratio(monkeypatch, caplog):
    pd = pytest.importorskip("pandas")

    symbol = "AAPL"
    ctx, state, feat_df = _build_dummy_long_context(pd, symbol)
    ctx.data_client = object()

    monkeypatch.setenv("EXECUTION_ALLOW_FALLBACK_PRICE", "0")
    reload_trading_config()

    _patch_price_gate_common(monkeypatch, symbol, 100.0)
    monkeypatch.setattr(
        bot_engine,
        "_fetch_quote",
        lambda *_a, **_k: types.SimpleNamespace(
            bid_price=110.0,
            ask_price=112.0,
            timestamp=datetime.now(UTC),
        ),
    )

    caplog.set_level("INFO")
    result = bot_engine._enter_long(
        ctx,
        state,
        symbol,
        balance=100000.0,
        feat_df=feat_df,
        final_score=1.0,
        conf=0.9,
        strat="gate_gap",
    )

    assert result is True
    assert any(
        record.message
        == "ORDER_SKIPPED_UNRELIABLE_PRICE | symbol=AAPL reason=gap_ratio>limit"
        for record in caplog.records
    )
    skip_record = next(
        record for record in caplog.records if record.message.startswith("ORDER_SKIPPED_UNRELIABLE_PRICE")
    )
    assert skip_record.reason == "gap_ratio>limit"
    expected_limit = float(getattr(bot_engine.get_trading_config(), "gap_ratio_limit", 0.0) or 0.0)
    assert skip_record.gap_limit == pytest.approx(expected_limit)
    assert not any("SIGNAL_BUY" in record.message for record in caplog.records)

    monkeypatch.delenv("EXECUTION_ALLOW_FALLBACK_PRICE", raising=False)
    reload_trading_config()


def _build_dummy_short_context(pd, symbol):
    class _DummyAPI:
        def get_asset(self, sym):
            assert sym == symbol
            return types.SimpleNamespace(shortable=True, shortable_shares=50)

        def get_account(self):
            return types.SimpleNamespace(
                equity=100000.0,
                portfolio_value=100000.0,
                shorting_power=50000.0,
                shorting_enabled=True,
            )

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
    ctx.allow_short_selling = True

    state = bot_engine.BotState()

    return ctx, state, feat_df


def test_enter_short_skips_when_shorting_unavailable(monkeypatch, caplog):
    pd = pytest.importorskip("pandas")

    symbol = "AAPL"
    ctx, state, feat_df = _build_dummy_short_context(pd, symbol)

    class _NoShortAPI:
        def get_asset(self, sym):
            return types.SimpleNamespace(shortable=True, shortable_shares=100)

        def get_account(self):
            return types.SimpleNamespace(
                equity=100000.0,
                portfolio_value=100000.0,
                shorting_power=0.0,
                shorting_enabled=False,
            )

    ctx.api = _NoShortAPI()

    monkeypatch.delenv("PYTEST_RUNNING", raising=False)
    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.delenv("DRY_RUN", raising=False)

    monkeypatch.setattr(bot_engine, "_resolve_order_quote", lambda *_a, **_k: (100.0, "alpaca_ask"))
    monkeypatch.setattr(bot_engine, "_apply_sector_cap_qty", lambda _ctx, _sym, qty, _price: qty)
    monkeypatch.setattr(bot_engine, "calculate_entry_size", lambda *_a, **_k: 10)
    monkeypatch.setattr(bot_engine, "scaled_atr_stop", lambda **_k: (95.0, 105.0))
    monkeypatch.setattr(bot_engine, "is_high_vol_regime", lambda: False)
    monkeypatch.setattr(bot_engine, "get_take_profit_factor", lambda: 1.0)
    monkeypatch.setattr(bot_engine, "_record_trade_in_frequency_tracker", lambda *a, **k: None)

    caplog.set_level("INFO")

    result = bot_engine._enter_short(
        ctx,
        state,
        symbol,
        feat_df=feat_df,
        final_score=-1.0,
        conf=-0.5,
        strat="short_guard",
    )

    assert result is True
    assert any(record.message == "SKIP_SHORTING_UNAVAILABLE" for record in caplog.records)


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


def test_enter_long_uses_fallback_when_env_flag_unset(monkeypatch, caplog):
    pd = pytest.importorskip("pandas")

    symbol = "AAPL"
    orders: list[tuple[str, int, str, float | None]] = []
    ctx, state, feat_df = _build_dummy_long_context(pd, symbol)

    monkeypatch.delenv("AI_TRADING_EXEC_ALLOW_FALLBACK_PRICE", raising=False)
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
        strat="fallback_env_unset_test",
    )

    assert result is True
    assert orders and orders[0][3] == pytest.approx(101.0)
    assert any(
        record.message == "ORDER_USING_FALLBACK_PRICE" for record in caplog.records
    )
    assert not any(
        record.message == "FALLBACK_PRICE_DISABLED" for record in caplog.records
    )


def test_enter_long_warns_when_fallback_quote_unavailable(monkeypatch, caplog):
    pd = pytest.importorskip("pandas")

    symbol = "AAPL"
    orders: list[tuple[str, int, str, float | None]] = []
    ctx, state, feat_df = _build_dummy_long_context(pd, symbol)
    ctx.data_client = None

    monkeypatch.setenv("AI_TRADING_STRICT_GATING", "1")
    bot_engine._strict_data_gating_enabled.cache_clear()

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
        strat="fallback_source_transient_warning",
    )

    assert result is True
    assert orders and orders[0][3] == pytest.approx(101.0)
    assert any(
        record.message == "DATA_GATING_PASS_THROUGH" for record in caplog.records
    )
    assert any(
        record.message == "ORDER_USING_FALLBACK_PRICE" for record in caplog.records
    )

    quality_meta = state.data_quality.get(symbol, {})
    assert "quote_source_unavailable" in quality_meta.get("gate_reasons", ())
    assert quality_meta.get("fallback_quote_error") == "quote_source_unavailable"


def test_enter_long_skips_when_only_last_close_available(monkeypatch, caplog):
    pd = pytest.importorskip("pandas")

    symbol = "AAPL"
    orders: list[tuple[str, int, str, float | None]] = []
    ctx, state, feat_df = _build_dummy_long_context(pd, symbol)

    monkeypatch.delenv("PYTEST_RUNNING", raising=False)
    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.delenv("DRY_RUN", raising=False)

    monkeypatch.setattr(
        bot_engine,
        "_evaluate_data_gating",
        lambda *a, **k: bot_engine.DataGateDecision(False, tuple(), None, {}),
    )
    monkeypatch.setattr(
        bot_engine,
        "submit_order",
        lambda _ctx, sym, qty, side, price=None: orders.append((sym, qty, side, price)),
    )
    monkeypatch.setattr(
        bot_engine,
        "_resolve_order_quote",
        lambda *_a, **_k: (101.0, "last_close"),
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
        strat="last_close_skip_test",
    )

    assert result is True
    assert orders == []
    skip_logs = [
        record
        for record in caplog.records
        if record.message.startswith("ORDER_SKIPPED_UNRELIABLE_PRICE")
    ]
    assert skip_logs, "Expected skip log for unreliable fallback price"
    assert any(
        getattr(record, "skip_reason", "") == "nbbo_missing_fallback_price"
        for record in skip_logs
    )
    assert not any(
        record.message == "ORDER_USING_FALLBACK_PRICE" for record in caplog.records
    )


def test_enter_long_blocks_on_stale_fallback_quote(monkeypatch, caplog):
    pd = pytest.importorskip("pandas")

    symbol = "AAPL"
    orders: list[tuple[str, int, str, float | None]] = []
    ctx, state, feat_df = _build_dummy_long_context(pd, symbol)

    monkeypatch.setenv("AI_TRADING_STRICT_GATING", "1")
    bot_engine._strict_data_gating_enabled.cache_clear()

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
    monkeypatch.setattr(
        bot_engine,
        "_check_fallback_quote_age",
        lambda *_a, **_k: (False, 86400.0, "fallback_quote_stale"),
    )
    monkeypatch.setattr(
        bot_engine,
        "_fallback_quote_newer_than_last_close",
        lambda *_a, **_k: False,
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
        strat="fallback_source_stale_block",
    )

    assert result is True
    assert orders == []
    assert any(
        record.message.startswith("ORDER_SKIPPED_UNRELIABLE_PRICE")
        for record in caplog.records
    )
    assert not any(
        record.message == "ORDER_USING_FALLBACK_PRICE" for record in caplog.records
    )

    quality_meta = state.data_quality.get(symbol, {})
    assert "fallback_quote_stale" in tuple(quality_meta.get("gate_reasons", ()))
    assert quality_meta.get("fallback_quote_error") == "fallback_quote_stale"


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


def test_enter_short_uses_fallback_when_env_flag_unset(monkeypatch, caplog):
    pd = pytest.importorskip("pandas")

    symbol = "AAPL"
    orders: list[tuple[str, int, str, float | None]] = []
    ctx, state, feat_df = _build_dummy_short_context(pd, symbol)

    monkeypatch.delenv("AI_TRADING_EXEC_ALLOW_FALLBACK_PRICE", raising=False)
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
        strat="fallback_short_env_unset_test",
    )

    assert result is True
    assert orders and orders[0][3] == pytest.approx(101.0)
    assert any(
        record.message == "ORDER_USING_FALLBACK_PRICE" for record in caplog.records
    )
    assert not any(
        record.message == "FALLBACK_PRICE_DISABLED" for record in caplog.records
    )


def test_enter_short_skips_when_only_last_close_available(monkeypatch, caplog):
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
        "_evaluate_data_gating",
        lambda *a, **k: bot_engine.DataGateDecision(False, tuple(), None, {}),
    )
    monkeypatch.setattr(
        bot_engine,
        "submit_order",
        lambda _ctx, sym, qty, side, price=None: orders.append((sym, qty, side, price)),
    )
    monkeypatch.setattr(
        bot_engine,
        "_resolve_order_quote",
        lambda *_a, **_k: (101.0, "last_close"),
    )

    caplog.set_level("INFO")

    result = bot_engine._enter_short(
        ctx,
        state,
        symbol,
        feat_df=feat_df,
        final_score=-1.0,
        conf=-0.8,
        strat="last_close_short_skip_test",
    )

    assert result is True
    assert orders == []
    skip_logs = [
        record
        for record in caplog.records
        if record.message.startswith("ORDER_SKIPPED_UNRELIABLE_PRICE")
    ]
    assert skip_logs, "Expected skip log for unreliable fallback price"
    assert any(
        getattr(record, "skip_reason", "") == "nbbo_missing_fallback_price"
        for record in skip_logs
    )
    assert not any(
        record.message == "ORDER_USING_FALLBACK_PRICE" for record in caplog.records
    )


def test_enter_short_skips_when_primary_disabled(monkeypatch, caplog):
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
        lambda *_a, **_k: (None, bot_engine._ALPACA_DISABLED_SENTINEL),
    )

    caplog.set_level("WARNING")

    result = bot_engine._enter_short(
        ctx,
        state,
        symbol,
        feat_df=feat_df,
        final_score=-1.0,
        conf=-0.8,
        strat="cooldown_short_skip",
    )

    assert result is True
    assert orders == []
    assert any(
        record.message == "SKIP_ORDER_ALPACA_UNAVAILABLE" for record in caplog.records
    )
