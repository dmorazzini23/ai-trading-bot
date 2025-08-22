# Ensure repository root in path
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.usefixtures("default_env")

# stub missing deps
req_mod = types.ModuleType("requests")
req_mod.post = lambda *a, **k: None
sys.modules.setdefault("requests", req_mod)
for _m in ["dotenv"]:
    mod = types.ModuleType(_m)
    if _m == "dotenv":
        mod.load_dotenv = lambda *a, **k: None
    sys.modules.setdefault(_m, mod)

try:
    from ai_trading import alpaca_api  # AI-AGENT-REF: canonical import
except (ValueError, TypeError):
    pytest.skip("alpaca_api not available", allow_module_level=True)
try:
    from ai_trading import rebalancer
except (ValueError, TypeError):
    pytest.skip("alpaca_trade_api not available", allow_module_level=True)
try:
    from ai_trading.execution import slippage  # AI-AGENT-REF: use prod slippage module
except (ValueError, TypeError):  # pragma: no cover - module optional
    slippage = None



def test_submit_order_shadow(monkeypatch):
    """submit_order returns a shadow status when SHADOW_MODE is enabled."""
    class DummyAPI:
        def submit_order(self, order_data=None):
            raise AssertionError("should not call in shadow")

    monkeypatch.setattr(alpaca_api, "SHADOW_MODE", True)
    log = types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    resp = alpaca_api.submit_order(
        DummyAPI(),
        types.SimpleNamespace(symbol="AAPL", qty=1, side="buy", time_in_force="day"),
        log,
    )
    assert resp["status"] == "shadow"


def test_monitor_slippage_alert(monkeypatch):
    """An alert is sent when slippage exceeds the threshold."""
    if not slippage or not hasattr(slippage, "monitor_slippage"):
        pytest.skip("slippage monitoring not available")
    called = []
    monkeypatch.setattr(slippage, "SLIPPAGE_THRESHOLD", 0.001)
    monkeypatch.setattr(slippage.logger, "warning", lambda m: called.append(m))
    slippage.monitor_slippage(100.0, 102.0, "AAPL")
    assert called


def test_maybe_rebalance(monkeypatch):
    """maybe_rebalance triggers a rebalance after the interval."""
    calls = []
    monkeypatch.setattr(rebalancer, "rebalance_interval_min", lambda: 0)
    monkeypatch.setattr(rebalancer, "rebalance_portfolio", lambda ctx: calls.append(ctx))
    rebalancer._last_rebalance = rebalancer.datetime.now(rebalancer.UTC) - rebalancer.timedelta(minutes=1)
    rebalancer.maybe_rebalance("ctx")
    assert calls == ["ctx"]


def test_atr_stop_adjusts():
    from ai_trading.risk.engine import (
        calculate_atr_stop,  # AI-AGENT-REF: normalized import
    )
    stop1 = calculate_atr_stop(100, 2, 1.5)
    stop2 = calculate_atr_stop(100, 5, 1.5)
    assert stop1 > stop2


def test_pyramiding_adds():
    from ai_trading.trade_logic import pyramiding_logic
    new_pos = pyramiding_logic(1, profit_in_atr=1.2, base_size=1)
    assert new_pos > 1


def test_volatility_filter():
    from ai_trading.meta_learning import volatility_regime_filter
    assert volatility_regime_filter(6, 100) == "high_vol"
    assert volatility_regime_filter(2, 100) == "low_vol"
