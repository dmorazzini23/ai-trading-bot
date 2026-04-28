from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from ai_trading.risk import circuit_breakers as cb
from ai_trading.risk import engine as risk_engine
from ai_trading.risk import manager as risk_manager
from ai_trading.risk.circuit_breakers import CircuitBreakerState


def _bare_engine(**config_updates: Any) -> risk_engine.RiskEngine:
    engine = object.__new__(risk_engine.RiskEngine)
    config = {
        "atr_multiplier": 2.0,
        "position_size_min_usd": 250.0,
        "max_symbol_exposure": 0.10,
        "min_order_value": 100.0,
        "max_order_value": 1_000.0,
        "max_concurrent_orders": 7,
        "order_spacing_seconds": 2.5,
    }
    config.update(config_updates)
    engine.config = SimpleNamespace(**config)
    engine.global_limit = 0.50
    engine.asset_limits = {}
    engine.strategy_limits = {}
    engine.exposure = {}
    engine.strategy_exposure = {}
    engine._positions = {}
    engine._returns = []
    engine._drawdowns = []
    engine._atr_cache = {}
    engine._volatility_cache = {}
    engine._volatility_alerted = False
    engine._last_portfolio_cap = None
    engine._last_equity_cap = None
    engine._invalid_min_size_logged = False
    engine._lock = threading.Lock()
    engine._update_event = threading.Event()
    engine._last_update = 0.0
    engine.current_trades = 0
    engine.max_trades = 2
    engine.hard_stop = False
    engine.max_drawdown_threshold = 0.15
    engine.hard_stop_recovery_threshold = 0.12
    engine.hard_stop_cooldown = 10.0
    engine._hard_stop_until = None
    engine._hard_stop_requires_manual_reset = False
    engine.enable_portfolio_features = True
    engine.data_client = None
    return cast(risk_engine.RiskEngine, engine)


def _signal(**updates: Any) -> risk_engine.TradeSignal:
    values = {
        "symbol": "AAPL",
        "side": "buy",
        "confidence": 0.8,
        "strategy": "momentum",
        "weight": 0.10,
        "asset_class": "equity",
    }
    values.update(updates)
    return risk_engine.TradeSignal(**values)


def test_engine_init_validates_invalid_config_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get_env(name: str, default: Any = None, *, cast: Any = None) -> Any:
        if name == "MAX_DRAWDOWN_THRESHOLD":
            return 2.0
        if name == "HARD_STOP_COOLDOWN_MIN":
            return -1.0
        return default

    monkeypatch.setattr(risk_engine.RiskEngine, "_validate_env", lambda self: None)
    monkeypatch.setattr(risk_engine, "get_settings", lambda: SimpleNamespace(ENABLE_PORTFOLIO_FEATURES=True))
    monkeypatch.setattr(risk_engine, "_resolve_alpaca_env", lambda: (None, None, None))
    monkeypatch.setattr(risk_engine, "get_env", fake_get_env)

    engine = risk_engine.RiskEngine(SimpleNamespace(exposure_cap_aggressive="bad"))

    assert engine.global_limit == pytest.approx(0.8)
    assert engine.max_drawdown_threshold == pytest.approx(0.15)
    assert engine.hard_stop_cooldown == pytest.approx(10.0)


def test_engine_init_raises_on_partial_alpaca_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get_env(name: str, default: Any = None, *, cast: Any = None) -> Any:
        if name == "ALPACA_API_KEY":
            return "key"
        if name == "ALPACA_SECRET_KEY":
            return None
        return default

    monkeypatch.setattr(risk_engine.RiskEngine, "_validate_env", lambda self: None)
    monkeypatch.setattr(risk_engine, "get_settings", lambda: SimpleNamespace(ENABLE_PORTFOLIO_FEATURES=False))
    monkeypatch.setattr(risk_engine, "_resolve_alpaca_env", lambda: (None, None, None))
    monkeypatch.setattr(risk_engine, "get_env", fake_get_env)

    with pytest.raises(RuntimeError, match="must both be provided"):
        risk_engine.RiskEngine(SimpleNamespace(exposure_cap_aggressive=0.4))


def test_safe_helpers_and_data_client_resolution() -> None:
    engine = _bare_engine()
    engine.data_client = object()
    assert engine._init_data_client() is engine.data_client

    ctx_api = object()
    engine.data_client = None
    engine.ctx = SimpleNamespace(api=ctx_api)
    assert engine._init_data_client() is ctx_api
    assert risk_engine._safe_call(lambda: "ok") == "ok"
    assert risk_engine._safe_call(lambda: (_ for _ in ()).throw(AttributeError("missing"))) is None
    assert risk_engine._is_finite_number(1.0) is True
    assert risk_engine._is_finite_number(float("nan")) is False
    assert risk_engine._is_finite_number(object()) is False


def test_is_finite_number_uses_float_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    class Floatable:
        def __float__(self) -> float:
            return 1.25

    monkeypatch.delattr(risk_engine.np, "isfinite", raising=False)

    assert risk_engine._is_finite_number(Floatable()) is True


def test_engine_init_env_parse_exceptions_and_config_attribute_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class BadConfig:
        def __getattribute__(self, name: str) -> Any:
            if name == "exposure_cap_aggressive":
                raise TypeError("bad exposure")
            return super().__getattribute__(name)

    calls: list[str] = []

    def fake_get_env(name: str, default: Any = None, *, cast: Any = None) -> Any:
        calls.append(name)
        if name in {"MAX_DRAWDOWN_THRESHOLD", "HARD_STOP_COOLDOWN_MIN"}:
            raise RuntimeError(f"{name} bad")
        return default

    monkeypatch.setattr(risk_engine.RiskEngine, "_validate_env", lambda self: None)
    monkeypatch.setattr(risk_engine, "get_settings", lambda: SimpleNamespace(ENABLE_PORTFOLIO_FEATURES=True))
    monkeypatch.setattr(risk_engine, "_resolve_alpaca_env", lambda: (None, None, None))
    monkeypatch.setattr(risk_engine, "get_env", fake_get_env)

    engine = risk_engine.RiskEngine(BadConfig())

    assert engine.global_limit == pytest.approx(0.8)
    assert engine.max_drawdown_threshold == pytest.approx(0.15)
    assert engine.hard_stop_cooldown == pytest.approx(10.0)
    assert "MAX_DRAWDOWN_THRESHOLD" in calls


def test_current_volatility_fallbacks_and_alert_suppression(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _bare_engine()
    assert engine._current_volatility() == pytest.approx(risk_engine.DEFAULT_VOLATILITY_FALLBACK)
    assert engine._volatility_alerted is True

    engine._volatility_alerted = False
    engine._returns = [0.0, 0.0, 0.0]
    assert engine._current_volatility() == pytest.approx(risk_engine.DEFAULT_VOLATILITY_FALLBACK)
    assert engine._volatility_alerted is True

    engine._volatility_alerted = False
    engine._returns = [0.01, -0.02, 0.03]
    monkeypatch.setattr(risk_engine.np, "std", lambda _values: (_ for _ in ()).throw(TypeError("bad std")))
    assert engine._current_volatility() == pytest.approx(risk_engine.DEFAULT_VOLATILITY_FALLBACK)


def test_current_volatility_success_clears_alert() -> None:
    engine = _bare_engine()
    engine._volatility_alerted = True
    engine._returns = [0.01, -0.02, 0.03]

    assert engine._current_volatility() > 0
    assert engine._volatility_alerted is False


def test_volatility_anomaly_logs_unmergeable_details(caplog: pytest.LogCaptureFixture) -> None:
    engine = _bare_engine()

    with caplog.at_level(logging.WARNING):
        engine._log_volatility_anomaly(
            "non_positive",
            fallback=0.02,
            details=cast(Any, [(1, 2, 3)]),
        )

    assert engine._volatility_alerted is True
    assert any("PORTFOLIO_VOLATILITY_FALLBACK" in record.message for record in caplog.records)


def test_atr_cache_dataframe_and_simple_get_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _bare_engine()
    engine._atr_cache["CACHED"] = (datetime.now(UTC), 1.23)
    assert engine._get_atr_data("CACHED") == pytest.approx(1.23)

    bars = pd.DataFrame(
        {
            "High": [11.0, 12.0, 13.0, 14.0],
            "Low": [9.0, 10.0, 11.0, 12.0],
            "Close": [10.0, 11.0, 12.0, 13.0],
        }
    )
    client = SimpleNamespace(get_stock_bars=lambda request: None)
    engine.data_client = client
    monkeypatch.setattr(risk_engine, "safe_get_stock_bars", lambda *_args, **_kwargs: bars)

    assert engine._get_atr_data("DF", lookback=3) == pytest.approx(2.0)

    class Client:
        def get_stock_bars(self, _request):
            return None

        def get_bars(self, _symbol, _limit):
            return [
                {"h": 10.0, "l": 8.0, "c": 9.0},
                {"h": 11.0, "l": 9.0, "c": 10.0},
                {"h": 12.0, "l": 10.0, "c": 11.0},
            ]

    engine = _bare_engine()
    engine.data_client = Client()
    monkeypatch.setattr(
        risk_engine,
        "safe_get_stock_bars",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("provider down")),
    )

    assert engine._get_atr_data("SEQ", lookback=3) == pytest.approx(2.0)


def test_atr_context_and_fetcher_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    class TypeErrorClient:
        def get_stock_bars(self, _request):
            return None

        def get_bars(self, _symbol, _limit):
            raise TypeError("legacy signature mismatch")

    engine = _bare_engine()
    engine.data_client = TypeErrorClient()
    engine.ctx = SimpleNamespace(
        minute_data={
            "CTX": pd.DataFrame(
                {
                    "high": [21.0, 22.0, 23.0],
                    "low": [19.0, 20.0, 21.0],
                    "close": [20.0, 21.0, 22.0],
                }
            )
        },
        daily_data={},
    )
    monkeypatch.setattr(risk_engine, "safe_get_stock_bars", lambda *_args, **_kwargs: None)

    assert engine._get_atr_data("CTX", lookback=3) == pytest.approx(2.0)

    class Fetcher:
        def get_daily_df(self, _ctx, _symbol):
            return pd.DataFrame({"High": [5.0, 6.0, 7.0], "Low": [4.0, 5.0, 6.0], "Close": [4.5, 5.5, 6.5]})

    engine = _bare_engine()
    engine.ctx = SimpleNamespace(minute_data={}, daily_data={}, data_fetcher=Fetcher())
    assert engine._get_atr_data("FETCH", lookback=3) == pytest.approx(1.3333333333333333)


def test_atr_sequence_objects_to_dict_and_insufficient_data(monkeypatch: pytest.MonkeyPatch) -> None:
    class RecordSource:
        def to_dict(self, orient: str):
            assert orient == "records"
            return [
                {"High": 10.0, "Low": 8.0, "Close": 9.0},
                {"High": 11.0, "Low": 9.0, "Close": 10.0},
                {"High": 12.0, "Low": 10.0, "Close": 11.0},
            ]

    class Client:
        def get_stock_bars(self, _request):
            return None

        def get_bars(self, _symbol, _limit):
            return RecordSource()

    engine = _bare_engine()
    engine.data_client = Client()
    monkeypatch.setattr(
        risk_engine,
        "safe_get_stock_bars",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("provider down")),
    )
    assert engine._get_atr_data("DICT", lookback=3) == pytest.approx(2.0)

    engine = _bare_engine()
    engine.ctx = SimpleNamespace(minute_data={"SHORT": pd.DataFrame({"high": [1.0], "low": [0.5], "close": [0.8]})}, daily_data={})
    assert engine._get_atr_data("SHORT", lookback=3) is None


def test_atr_sequence_bar_object_and_tuple_records(monkeypatch: pytest.MonkeyPatch) -> None:
    class Client:
        def __init__(self, bars: Any) -> None:
            self.bars = bars

        def get_stock_bars(self, _request):
            return None

        def get_bars(self, _symbol, _limit):
            return self.bars

    monkeypatch.setattr(
        risk_engine,
        "safe_get_stock_bars",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("provider down")),
    )
    engine = _bare_engine()
    engine.data_client = Client(
        [
            SimpleNamespace(high=10.0, low=8.0, close=9.0),
            SimpleNamespace(high=11.0, low=9.0, close=10.0),
            SimpleNamespace(high=12.0, low=10.0, close=11.0),
        ]
    )
    assert engine._get_atr_data("OBJ", lookback=3) == pytest.approx(2.0)

    engine = _bare_engine()
    engine.data_client = Client([(1.0, 10.0, 8.0, 9.0), (1.0, 11.0, 9.0, 10.0), (1.0, 12.0, 10.0, 11.0)])
    assert engine._get_atr_data("TUPLE4", lookback=3) == pytest.approx(2.0)

    engine = _bare_engine()
    engine.data_client = Client([(10.0, 8.0, 9.0), (11.0, 9.0, 10.0), (12.0, 10.0, 11.0)])
    assert engine._get_atr_data("TUPLE3", lookback=3) == pytest.approx(2.0)


def test_adaptive_caps_available_exposure_and_portfolio_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _bare_engine(exposure_cap_conservative=0.2, volatility_lookback_days=5)
    engine._returns = [0.03, 0.02, 0.04, 0.01, 0.03]
    assert engine._adaptive_global_cap() > engine.global_limit

    engine._returns = [-0.05, -0.04, -0.03, -0.02, -0.01]
    assert engine._adaptive_global_cap() < engine.global_limit

    engine.exposure = {"equity": 0.2, "crypto": "bad"}
    monkeypatch.setattr(engine, "_adaptive_global_cap", lambda: (_ for _ in ()).throw(TypeError("bad cap")))
    assert engine.available_exposure() == pytest.approx(0.3)
    assert engine.available_exposure(cap="bad") == pytest.approx(0.3)

    engine = _bare_engine()
    engine.max_drawdown_threshold = 0.1
    engine.enable_portfolio_features = False
    engine.update_portfolio_metrics([0.01], 0.2)
    assert engine._returns == []
    engine.enable_portfolio_features = True
    engine.update_portfolio_metrics([0.01], 0.2)
    assert engine._returns == [0.01]
    assert engine.hard_stop is True


def test_refresh_position_lookup_and_update_exposure_error_paths() -> None:
    engine = _bare_engine()

    class API:
        def list_positions(self):
            return [
                SimpleNamespace(asset_class="equity", qty="10", avg_entry_price="100", symbol="AAPL"),
                SimpleNamespace(asset_class="crypto", qty="2", avg_entry_price="50", symbol="BTCUSD"),
            ]

        def get_account(self):
            return SimpleNamespace(equity="2000")

    api = API()
    engine.refresh_positions(api)
    assert engine.exposure == {"equity": pytest.approx(0.5), "crypto": pytest.approx(0.05)}
    assert engine.position_exists(api, "AAPL") is True
    assert engine.position_exists(api, "MSFT") is False

    with pytest.raises(RuntimeError):
        engine.update_exposure()

    engine.update_exposure(SimpleNamespace(api=SimpleNamespace(list_positions=lambda: (_ for _ in ()).throw(AttributeError("missing")))))


def test_refresh_positions_counts_short_exposure_as_gross() -> None:
    engine = _bare_engine()

    class API:
        def get_all_positions(self):
            return [
                SimpleNamespace(asset_class="equity", qty="-5", avg_entry_price="100", symbol="AAPL"),
                SimpleNamespace(asset_class="equity", qty="3", avg_entry_price="50", symbol="MSFT"),
            ]

        def get_account(self):
            return SimpleNamespace(equity="1000")

    engine.refresh_positions(API())

    assert engine.exposure == {"equity": pytest.approx(0.65)}


def test_can_trade_rejects_limits_and_allows_force_override(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _bare_engine()
    assert engine.can_trade(object()) is False

    engine.exposure = {"equity": 0.50}
    rejected = _signal(weight=0.20)
    monkeypatch.setattr(risk_engine, "get_env", lambda *_args, **_kwargs: False)
    assert engine.can_trade(rejected) is False

    engine.exposure = {}
    engine.strategy_limits = {"momentum": 0.05}
    assert engine.can_trade(_signal(weight=0.20)) is False

    monkeypatch.setattr(risk_engine, "get_env", lambda *_args, **_kwargs: True)
    assert engine.can_trade(_signal(weight=0.20)) is True

    engine = _bare_engine()
    assert engine.can_trade(_signal(weight="bad")) is True


def test_can_trade_keeps_expired_hard_stop_until_fresh_recovered_drawdown() -> None:
    engine = _bare_engine()
    engine.hard_stop = True
    engine._hard_stop_until = 0.0

    assert engine.can_trade(_signal(), returns=[0.01], drawdowns=["bad"]) is False
    assert engine.hard_stop is True
    assert engine._returns[-1] == 0.01

    assert engine.can_trade(_signal(), drawdowns=[0.13]) is False
    assert engine.hard_stop is True

    assert engine.can_trade(_signal(), drawdowns=[0.11]) is True
    assert engine.hard_stop is False


def test_register_fill_and_trade_slot_state_updates() -> None:
    engine = _bare_engine()
    engine.register_fill(object())

    engine.exposure = {"equity": 0.05}
    engine.strategy_exposure = {"momentum": 0.05}
    engine.register_fill(_signal(side="sell", weight=0.20))
    assert engine.exposure["equity"] == 0.0
    assert engine._update_event.is_set()

    engine.register_fill(_signal(side="buy", weight="bad"))
    assert engine.exposure["equity"] == 0.0
    engine.wait_for_exposure_update(timeout=0.01)
    assert not engine._update_event.is_set()

    engine.update_position("AAPL", 3, "buy")
    engine.update_position("AAPL", 1, "sell")
    assert engine._positions["AAPL"] == 2

    for value in range(95):
        engine.update_returns(float(value))
    assert len(engine._returns) == 90
    assert engine._returns[0] == 5.0


def test_trade_slots_and_explicit_hard_stop_with_missing_lock() -> None:
    engine = _bare_engine()
    engine._lock = None
    assert engine.acquire_trade_slot() is True
    assert engine.acquire_trade_slot() is True
    assert engine.acquire_trade_slot() is False
    engine.release_trade_slot()
    engine.release_trade_slot()
    engine.release_trade_slot()
    assert engine.current_trades == 0
    engine.trigger_hard_stop()
    assert engine.hard_stop is True
    engine._maybe_lift_hard_stop()
    assert engine.hard_stop is True
    assert engine.reset_hard_stop(current_drawdown=0.20) is False
    assert engine.hard_stop is True
    assert engine.reset_hard_stop(current_drawdown=0.05) is True
    assert engine.hard_stop is False


def test_apply_risk_scaling_uses_volatility_and_cvar(monkeypatch: pytest.MonkeyPatch) -> None:
    import ai_trading.capital_scaling as capital_scaling

    monkeypatch.setattr(capital_scaling, "cvar_scaling", lambda _arr, alpha: 2.0)
    engine = _bare_engine()
    signal = _signal(weight=0.60)

    scaled = engine.apply_risk_scaling(signal, volatility=0.04, returns=[-0.10, 0.01, 0.02])

    assert scaled.weight == pytest.approx(0.10)
    bad = SimpleNamespace(weight="bad")
    assert engine.apply_risk_scaling(bad).weight == "bad"


def test_check_max_drawdown_and_position_size_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _bare_engine()
    assert engine.check_max_drawdown(SimpleNamespace(get_account=lambda: SimpleNamespace(equity="90", last_equity="100"))) is False
    assert engine.hard_stop is True

    engine = _bare_engine()
    monkeypatch.setattr(engine, "can_trade", lambda _signal: False)
    assert engine.position_size(_signal(), 1_000.0, 100.0) == 0

    monkeypatch.setattr(engine, "can_trade", lambda _signal: True)
    assert engine.position_size(_signal(), 1_000.0, 0.0) == 0
    assert engine.position_size(_signal(), 0.0, 100.0) == 0

    assert engine.position_size(SimpleNamespace(weight=0.1), 1_000.0, 100.0) == 0


def test_position_size_atr_fallback_and_final_error(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _bare_engine(position_size_min_usd=100.0)
    monkeypatch.setattr(engine, "can_trade", lambda _signal: True)
    monkeypatch.setattr(engine, "_get_atr_data", lambda _symbol: 2.0)
    assert engine.position_size(_signal(), 10_000.0, 100.0) == 25

    monkeypatch.setattr(engine, "_get_atr_data", lambda _symbol: (_ for _ in ()).throw(ValueError("atr bad")))
    monkeypatch.setattr(engine, "_apply_weight_limits", lambda _signal: 0.10)
    assert engine.position_size(_signal(strategy="default"), 1_000.0, 100.0) == 5

    monkeypatch.setattr(risk_engine, "_calculate_position_size", lambda *_args: (_ for _ in ()).throw(ValueError("qty bad")))
    assert engine.position_size(_signal(), 1_000.0, 100.0) == 0

    engine = _bare_engine()
    monkeypatch.setattr(engine, "can_trade", lambda _signal: True)
    monkeypatch.setattr(engine, "_get_atr_data", lambda _symbol: (_ for _ in ()).throw(ValueError("atr bad")))
    monkeypatch.setattr(engine, "_apply_weight_limits", lambda _signal: (_ for _ in ()).throw(ValueError("weight bad")))
    assert engine.position_size(_signal(), 1_000.0, 100.0) == 0


def test_position_size_uses_account_fallback_and_rejects_bad_equity(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _bare_engine(position_size_min_usd=100.0)
    monkeypatch.setattr(engine, "can_trade", lambda _signal: True)
    monkeypatch.setattr(engine, "_get_atr_data", lambda _symbol: None)
    monkeypatch.setattr(engine, "_apply_weight_limits", lambda _signal: 0.10)

    class AccountFallbackAPI:
        def __init__(self) -> None:
            self.calls = 0

        def get_account(self):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(equity="1000", last_equity="1000")
            raise ValueError("account bad")

    api_error = AccountFallbackAPI()
    assert engine.position_size(_signal(), 1_000.0, 100.0, api=api_error) == 1

    api_zero = SimpleNamespace(get_account=lambda: SimpleNamespace(equity="0"))
    assert engine.position_size(_signal(), 1_000.0, 100.0, api=api_zero) == 0

    api_drawdown = SimpleNamespace(get_account=lambda: SimpleNamespace(equity="100", last_equity="200"))
    assert engine.position_size(_signal(), 1_000.0, 100.0, api=api_drawdown) == 0


def test_apply_weight_limits_and_compute_volatility_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _bare_engine()
    assert engine._apply_weight_limits(SimpleNamespace(asset_class="equity")) == 0.0
    assert engine._apply_weight_limits(_signal(weight="bad")) == 0.0

    class BadSignal:
        def __getattribute__(self, name: str) -> Any:
            if name == "asset_class":
                raise ValueError("bad signal")
            return super().__getattribute__(name)

    assert engine._apply_weight_limits(BadSignal()) == 0.0
    assert engine.compute_volatility(np.array([0.01, np.nan])) == {"volatility": 0.0, "mad": 0.0, "garch_vol": 0.0}

    class MinimalNP:
        def asarray(self, values):
            return list(values)

    monkeypatch.setattr(risk_engine, "np", MinimalNP())
    result = engine.compute_volatility([1.0, 3.0, 5.0])
    assert result["std_vol"] == pytest.approx(1.632993161855452)
    assert result["mad"] == pytest.approx(2.0)
    assert result["garch_vol"] > 0
    assert engine.compute_volatility(["nan"]) == {"volatility": 0.0, "mad": 0.0, "garch_vol": 0.0}


def test_position_and_order_limit_gates() -> None:
    engine = _bare_engine()
    engine.exposure = {"AAPL": 0.05, "MSFT": 0.40}
    assert engine.check_position_limits("AAPL", 40) is True
    assert engine.check_position_limits("AAPL", 100) is False
    engine.config.max_symbol_exposure = 1.0
    assert engine.check_position_limits("TSLA", 200) is False
    engine.exposure = {"AAPL": object()}
    assert engine.check_position_limits("AAPL", 1) is False

    assert engine.validate_order_size("AAPL", 2, 100.0) is True
    assert engine.validate_order_size("AAPL", 0.5, 100.0) is False
    assert engine.validate_order_size("AAPL", 0.5, 300.0) is False
    assert engine.validate_order_size("AAPL", 20, 100.0) is False
    assert engine.validate_order_size("AAPL", 20_000, 0.01) is True
    assert engine.validate_order_size("AAPL", 1, object()) is False
    assert engine.max_concurrent_orders() == 7
    assert engine.max_exposure() == pytest.approx(0.5)
    assert engine.order_spacing() == pytest.approx(2.5)
    assert engine.get_current_exposure() == engine.exposure


def test_module_position_sizing_and_drawdown_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeEngine:
        def __init__(self, _cfg):
            pass

        def position_size(self, signal, cash, price, api=None):
            assert cash == 1_000.0
            assert price == 100.0
            assert hasattr(signal, "symbol")
            return 42

    monkeypatch.setattr(risk_engine, "RiskEngine", FakeEngine)
    monkeypatch.setattr(risk_engine, "get_trading_config", lambda: object())

    assert risk_engine.calculate_position_size(1_000.0, 100.0) == 42
    assert risk_engine.calculate_position_size(_signal(), 1_000.0, 100.0) == 42
    assert risk_engine.calculate_position_size(-1.0, 100.0) == 0
    assert risk_engine.calculate_position_size(1_000.0, -1.0) == 0
    assert risk_engine.calculate_position_size(_signal(), -1.0, 100.0) == 0
    assert risk_engine.calculate_position_size(_signal(), 1_000.0, -1.0) == 0
    assert risk_engine.calculate_position_size(SimpleNamespace(), 1_000.0, 100.0) == 0

    with pytest.raises(TypeError):
        risk_engine.calculate_position_size(1.0)

    assert risk_engine.check_max_drawdown({"current_drawdown": 0.2, "max_drawdown": 0.1}) is True
    assert risk_engine.check_max_drawdown({"current_drawdown": 0.1, "max_drawdown": 0.2}) is False
    assert risk_engine.check_max_drawdown([]) is False
    assert risk_engine.check_max_drawdown({"current_drawdown": -1, "max_drawdown": 0.2}) is False
    assert risk_engine.check_max_drawdown({"current_drawdown": 0.1, "max_drawdown": 0}) is False


def test_standalone_risk_helpers_and_trailing_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    assert risk_engine.dynamic_position_size(0, 0.2, 0.0) == 0.0
    assert risk_engine.dynamic_position_size(1_000, 0.2, 0.2) == pytest.approx(500.0)
    assert risk_engine.can_trade(SimpleNamespace(hard_stop=False, current_trades=1, max_trades=2)) is True
    assert risk_engine.can_trade(SimpleNamespace(hard_stop=True, current_trades=0, max_trades=2)) is False
    assert risk_engine.register_trade(SimpleNamespace(acquire_trade_slot=lambda: True), 10) == {"size": 10}
    assert risk_engine.register_trade(SimpleNamespace(acquire_trade_slot=lambda: False), 10) is None
    assert risk_engine.register_trade(SimpleNamespace(acquire_trade_slot=lambda: True), 0) is None

    portfolio = SimpleNamespace(positions={"AAPL": SimpleNamespace(quantity=1), "MSFT": SimpleNamespace(quantity=0)})
    assert risk_engine.check_exposure_caps(portfolio, {"AAPL": 0.6, "MSFT": 0.8}, 0.5) is False
    assert risk_engine.check_exposure_caps(portfolio, {"AAPL": 0.1, "MSFT": 0.8}, 0.5) is None

    df = pd.DataFrame({"High": [10.0, 10.0, 10.0], "Low": [9.0, 9.0, 9.0], "Close": [7.0, 7.0, 7.0]})
    scheduled: list[tuple[str, int]] = []
    import ai_trading.indicators as indicators

    monkeypatch.setattr(risk_engine, "load_pandas_ta", lambda: None)
    monkeypatch.setattr(indicators, "atr", lambda *_args: pd.Series([1.0, 1.0, 1.0]))
    monkeypatch.setattr(risk_engine, "schedule_reentry_check", lambda symbol, lookahead_days: scheduled.append((symbol, lookahead_days)))
    risk_engine.apply_trailing_atr_stop(df, 10.0, symbol="AAPL")
    assert scheduled == [("AAPL", 2)]

    risk_engine.apply_trailing_atr_stop(df, -1.0, symbol="AAPL")
    nan_df = pd.DataFrame({"High": [10.0], "Low": [9.0], "Close": [float("nan")]})
    risk_engine.apply_trailing_atr_stop(nan_df, 10.0, symbol="AAPL")


def test_trailing_stop_sends_exit_when_position_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame({"High": [10.0, 10.0, 10.0], "Low": [9.0, 9.0, 9.0], "Close": [7.0, 7.0, 7.0]})
    sent: list[tuple[Any, ...]] = []
    bot_mod = SimpleNamespace(send_exit_order=lambda *args: sent.append(args))
    context = SimpleNamespace(
        api=object(),
        risk_engine=SimpleNamespace(position_exists=lambda _api, _symbol: True),
    )
    import ai_trading.indicators as indicators

    monkeypatch.setattr(risk_engine, "load_pandas_ta", lambda: None)
    monkeypatch.setattr(indicators, "atr", lambda *_args: pd.Series([1.0, 1.0, 1.0]))
    monkeypatch.setattr(risk_engine.importlib, "import_module", lambda _name: bot_mod)
    monkeypatch.setattr(risk_engine, "schedule_reentry_check", lambda *_args, **_kwargs: None)

    risk_engine.apply_trailing_atr_stop(df, 10.0, context=context, symbol="AAPL", qty=-3)

    assert sent == [(context, "AAPL", 3, 7.0, "atr_stop")]


def test_trailing_stop_skips_missing_position_and_handles_exit_error(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame({"High": [10.0, 10.0, 10.0], "Low": [9.0, 9.0, 9.0], "Close": [7.0, 7.0, 7.0]})
    import ai_trading.indicators as indicators

    monkeypatch.setattr(risk_engine, "load_pandas_ta", lambda: None)
    monkeypatch.setattr(indicators, "atr", lambda *_args: pd.Series([1.0, 1.0, 1.0]))
    monkeypatch.setattr(risk_engine, "schedule_reentry_check", lambda *_args, **_kwargs: None)
    missing_context = SimpleNamespace(
        api=object(),
        risk_engine=SimpleNamespace(position_exists=lambda _api, _symbol: False),
    )
    risk_engine.apply_trailing_atr_stop(df, 10.0, context=missing_context, symbol="AAPL", qty=3)

    error_context = SimpleNamespace(api=object())
    monkeypatch.setattr(risk_engine.importlib, "import_module", lambda _name: (_ for _ in ()).throw(ValueError("send bad")))
    risk_engine.apply_trailing_atr_stop(df, 10.0, context=error_context, symbol="AAPL", qty=3)
    risk_engine.apply_trailing_atr_stop(pd.DataFrame({"High": ["bad"], "Low": [1.0], "Close": [1.0]}), 10.0)


def test_stop_level_correlation_and_filter_helpers() -> None:
    corr = pd.DataFrame({"AAPL": [1.0, 0.5], "MSFT": [0.5, 1.0]})
    weights = risk_engine.correlation_position_weights(corr, {"AAPL": 0.6, "GOOG": 0.4})
    assert weights["AAPL"] == pytest.approx(0.6 / 1.75)
    assert weights["GOOG"] == pytest.approx(0.4)
    assert risk_engine.compute_stop_levels(100.0, 2.0) == (98.0, 104.0)
    assert risk_engine.calculate_bollinger_stop(100.0, 110.0, 90.0, direction="long") == 100.0
    assert risk_engine.calculate_bollinger_stop(95.0, 110.0, 90.0, direction="short") == 100.0
    assert risk_engine.dynamic_stop_price(100.0, upper_band=110.0, lower_band=90.0) == 100.0
    assert risk_engine.dynamic_stop_price(100.0) == 100.0
    assert risk_engine.drawdown_circuit([-0.05, -0.25], limit=0.2) is True
    assert risk_engine.drawdown_circuit([], limit=0.2) is False
    assert risk_engine.volatility_filter(2.0, 0.0) is True
    assert risk_engine.volatility_filter(2.0, 100.0, threshold=0.05) is True
    assert risk_engine.volatility_filter(10.0, 100.0, threshold=0.05) is False


def test_circuit_breaker_error_and_recovery_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    drawdown = cb.DrawdownCircuitBreaker(max_drawdown=0.1)
    drawdown.peak_equity = 100.0
    drawdown.max_drawdown = "bad"  # type: ignore[assignment]
    assert drawdown.update_equity(80.0) is False

    drawdown = cb.DrawdownCircuitBreaker(max_drawdown=0.1)
    assert "object" in drawdown._safe_format_percentage(object())
    monkeypatch.setattr(cb, "safe_utcnow", lambda: (_ for _ in ()).throw(ValueError("clock bad")))
    drawdown._trigger_halt("bad clock")
    assert drawdown.state is CircuitBreakerState.OPEN

    drawdown.reset_callbacks.append(lambda *_args: None)
    drawdown._reset_breaker("bad clock")

    from ai_trading.utils.time import safe_utcnow as real_safe_utcnow

    monkeypatch.setattr(cb, "safe_utcnow", real_safe_utcnow)
    volatility = cb.VolatilityCircuitBreaker(high_vol_threshold=0.5, extreme_vol_threshold=cast(Any, "bad"))
    assert volatility.update_volatility(0.6)["status"] == "ERROR"
    assert "object" in volatility._safe_format_percentage(object())

    drawdown = cb.DrawdownCircuitBreaker(max_drawdown=0.1)
    monkeypatch.setattr(cb.logger, "info", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("logger bad")))
    drawdown._reset_breaker("logger bad")


def test_trading_halt_manager_outer_error_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    class BadLock:
        def __enter__(self):
            raise ValueError("lock bad")

        def __exit__(self, *_args):
            return False

    manager = cb.TradingHaltManager()
    manager._lock = BadLock()  # type: ignore[assignment]

    assert manager.is_trading_allowed()["trading_allowed"] is False
    manager.update_equity(100.0)
    assert manager.update_volatility(0.2)["status"] == "ERROR"
    manager.manual_halt_trading("ops")
    manager.resume_trading()
    manager.emergency_stop_all("panic")
    manager.reset_emergency_stop()
    manager.record_trade(-1.0)
    manager.reset_daily_counters()
    assert "error" in manager.get_comprehensive_status()


def test_dead_mans_switch_error_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    switch = cb.DeadMansSwitch(timeout_seconds=1)
    monkeypatch.setattr(cb.threading, "Thread", lambda **_kwargs: (_ for _ in ()).throw(ValueError("thread bad")))
    switch.start_monitoring()

    switch = cb.DeadMansSwitch(timeout_seconds=1)
    switch.is_active = True
    switch.monitoring_thread = SimpleNamespace(is_alive=lambda: (_ for _ in ()).throw(ValueError("alive bad")))
    switch.stop_monitoring()

    switch = cb.DeadMansSwitch(timeout_seconds=1)
    switch.stop_monitoring()

    switch = cb.DeadMansSwitch(timeout_seconds=1)
    switch.is_active = True
    monkeypatch.setattr(cb, "safe_utcnow", lambda: (_ for _ in ()).throw(ValueError("clock bad")))
    switch.heartbeat()
    switch.get_status()
    from ai_trading.utils.time import safe_utcnow as real_safe_utcnow

    monkeypatch.setattr(cb, "safe_utcnow", real_safe_utcnow)

    switch = cb.DeadMansSwitch(timeout_seconds=1)
    switch.is_active = True
    switch._stop_event = SimpleNamespace(is_set=lambda: (_ for _ in ()).throw(ValueError("stop bad")), wait=lambda _seconds: None)
    switch._monitoring_loop()

    switch = cb.DeadMansSwitch(timeout_seconds=100)
    waits: list[float] = []

    class WaitOnce:
        def is_set(self) -> bool:
            return False

        def wait(self, seconds: float) -> None:
            waits.append(seconds)
            switch.is_active = False

    switch.is_active = True
    switch._stop_event = WaitOnce()  # type: ignore[assignment]
    switch._monitoring_loop()
    assert waits == [10.0]

    switch = cb.DeadMansSwitch(timeout_seconds=1)
    class BadCallbacks:
        def __iter__(self):
            raise ValueError("iter bad")

    switch.emergency_callbacks = cast(Any, BadCallbacks())
    switch._trigger_emergency()


def test_risk_manager_low_risk_alert_and_stress_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(risk_manager, "get_settings", lambda: SimpleNamespace(ENABLE_PORTFOLIO_FEATURES=True))
    manager = risk_manager.RiskManager()
    positions = [
        {"symbol": f"S{i}", "market_value": 1.0, "sector": f"Sector{i}"}
        for i in range(40)
    ]
    low = manager.check_portfolio_risk(
        positions,
        {},
    )
    assert low["overall_risk_level"] in {"Low", "Medium"}

    manager.add_risk_alert("portfolio", "watch", "medium")
    assert manager.get_risk_alerts()[0]["message"] == "watch"
    manager.update_drawdown(95.0, 100.0)
    assert manager.current_drawdown == pytest.approx(0.05)

    assessor = risk_manager.PortfolioRiskAssessor()
    stress = assessor._apply_stress_scenario(
        [{"symbol": "A", "market_value": 100.0}],
        {"A": -0.25, "market_shock": -0.10},
    )
    assert stress["portfolio_value_after"] == pytest.approx(75.0)
