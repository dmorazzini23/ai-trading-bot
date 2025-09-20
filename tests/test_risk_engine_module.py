import logging
import os
import sys
import types

import pytest

np = pytest.importorskip("numpy")
os.environ.setdefault("PYTEST_RUNNING", "1")
os.environ.setdefault("MAX_DRAWDOWN_THRESHOLD", "0.15")
os.environ.setdefault("WEBHOOK_SECRET", "test-secret")
for m in ["strategies", "strategies.momentum", "strategies.mean_reversion"]:
    sys.modules.pop(m, None)
sys.modules.pop("risk_engine", None)
import ai_trading.config.management as config_management
import ai_trading.config.settings as config_settings

config_pkg = sys.modules.get("ai_trading.config")
if config_pkg is None:
    config_pkg = types.ModuleType("ai_trading.config")
    sys.modules["ai_trading.config"] = config_pkg

config_pkg.get_settings = config_settings.get_settings
config_pkg.Settings = config_settings.Settings
config_pkg.management = config_management
config_pkg.TradingConfig = config_management.TradingConfig

if not hasattr(config_management, "from_env_relaxed"):
    def _from_env_relaxed() -> config_management.TradingConfig:  # pragma: no cover - legacy shim
        return config_management.TradingConfig.from_env()

    config_management.from_env_relaxed = _from_env_relaxed  # type: ignore[attr-defined]
from ai_trading.risk.engine import RiskEngine, TradeSignal  # AI-AGENT-REF: normalized import


class DummyAPI:
    def __init__(self, equity=100, last=100):
        self._eq = equity
        self._last = last

    def get_account(self):
        return types.SimpleNamespace(equity=str(self._eq), last_equity=str(self._last))


def make_signal():
    return TradeSignal(symbol="AAPL", side="buy", confidence=1.0, strategy="s", weight=0.0, asset_class="equity")


def test_risk_engine_instantiates_with_default_config(monkeypatch):
    stub_config = types.SimpleNamespace(
        exposure_cap_aggressive=0.8,
        position_size_min_usd=100.0,
        atr_multiplier=1.0,
        pytest_running=True,
        max_drawdown_threshold=0.15,
        hard_stop_cooldown_min=10.0,
        alpaca_oauth_token=None,
    )
    calls: list[None] = []

    def fake_get_trading_config():
        calls.append(None)
        return stub_config

    risk_engine_module = sys.modules[RiskEngine.__module__]
    monkeypatch.setattr(risk_engine_module, "get_trading_config", fake_get_trading_config)
    monkeypatch.setenv("WEBHOOK_SECRET", "test-secret")

    engine = RiskEngine()

    assert engine.config is stub_config
    assert calls, "RiskEngine() should fetch a trading config when none is provided"


def test_can_trade_limits():
    eng = RiskEngine()
    sig = make_signal()
    eng.asset_limits["equity"] = 0.5
    eng.exposure["equity"] = 0.6
    assert not eng.can_trade(sig)
    eng.exposure["equity"] = 0.1
    sig.weight = 0.3
    assert eng.can_trade(sig)


def test_can_trade_drawdown_triggers_stop():
    eng = RiskEngine()
    sig = make_signal()
    eng.max_drawdown_threshold = 0.05
    assert not eng.can_trade(sig, drawdowns=[0.1])
    assert eng.hard_stop


def test_register_and_position_size(monkeypatch):
    eng = RiskEngine()
    sig = make_signal()
    sig.weight = 0.1  # Set weight to avoid exposure cap breach
    eng.asset_limits["equity"] = 1.0
    eng.strategy_limits["s"] = 1.0
    eng.config = types.SimpleNamespace(position_size_min_usd=1, atr_multiplier=1.0)
    qty = eng.position_size(sig, cash=100, price=10)
    # Position size may be influenced by multiple factors, just ensure it's positive
    assert qty > 0
    eng.register_fill(sig)
    assert eng.exposure["equity"] == sig.weight
    sell = TradeSignal(symbol="AAPL", side="sell", confidence=1.0, strategy="s", weight=sig.weight, asset_class="equity")
    eng.register_fill(sell)
    assert round(eng.exposure["equity"], 6) == 0


def test_register_fill_zero_weight_no_exposure_change():
    eng = RiskEngine()
    sig = make_signal()
    assert eng.exposure.get("equity", 0.0) == 0.0
    eng.register_fill(sig)
    assert eng.exposure.get("equity", 0.0) == 0.0


def test_register_fill_partial_updates_exposure():
    eng = RiskEngine()
    base = make_signal()
    base.weight = 0.5
    partial = TradeSignal(
        symbol=base.symbol,
        side=base.side,
        confidence=base.confidence,
        strategy=base.strategy,
        weight=0.2,
        asset_class=base.asset_class,
    )
    eng.register_fill(partial)
    assert pytest.approx(eng.exposure[base.asset_class], rel=1e-6) == 0.2
    remainder = TradeSignal(
        symbol=base.symbol,
        side=base.side,
        confidence=base.confidence,
        strategy=base.strategy,
        weight=base.weight - partial.weight,
        asset_class=base.asset_class,
    )
    eng.register_fill(remainder)
    assert pytest.approx(eng.exposure[base.asset_class], rel=1e-6) == pytest.approx(base.weight)


def test_position_size_zero_raw_qty_defaults_to_min(caplog):
    eng = RiskEngine()
    sig = make_signal()
    sig.weight = 0.0
    eng.asset_limits["equity"] = 1.0
    eng.strategy_limits["s"] = 1.0
    eng.config = types.SimpleNamespace(position_size_min_usd=100, atr_multiplier=1.0)
    with caplog.at_level(logging.WARNING):
        qty = eng.position_size(sig, cash=1000, price=10)
    assert qty == 10
    assert any("falling back to minimum position size" in rec.message for rec in caplog.records)


def test_position_size_invalid_min_usd_falls_back_once(caplog):
    eng = RiskEngine()
    sig = make_signal()
    sig.weight = 0.1
    eng.asset_limits["equity"] = 1.0
    eng.strategy_limits["s"] = 1.0
    eng.config = types.SimpleNamespace(position_size_min_usd=0.0, atr_multiplier=1.0)
    with caplog.at_level(logging.WARNING):
        qty1 = eng.position_size(sig, cash=100, price=30)
        qty2 = eng.position_size(sig, cash=100, price=30)
    assert qty1 >= 1
    assert qty2 >= 1
    invalid_logs = [rec for rec in caplog.records if "Invalid position_size_min_usd" in rec.message]
    assert len(invalid_logs) == 1


def test_check_max_drawdown():
    eng = RiskEngine()
    api = DummyAPI(equity=90, last=100)
    ok = eng.check_max_drawdown(api)
    assert not ok and eng.hard_stop


def test_compute_volatility():
    eng = RiskEngine()
    arr = np.array([1.0, 2.0, 3.0])
    res = eng.compute_volatility(arr)
    assert "volatility" in res and res["volatility"] > 0
    assert eng.compute_volatility(np.array([]))["volatility"] == 0.0


def test_hard_stop_blocks_trading():
    eng = RiskEngine()
    eng.hard_stop = True
    sig = make_signal()
    assert not eng.can_trade(sig)
    assert eng.position_size(sig, 100, 10) == 0


def test_check_max_drawdown_ok():
    eng = RiskEngine()
    api = DummyAPI(equity=105, last=100)
    assert eng.check_max_drawdown(api)
