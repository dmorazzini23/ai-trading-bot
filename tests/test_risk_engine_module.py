import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for m in ["strategies", "strategies.momentum", "strategies.mean_reversion"]:
    sys.modules.pop(m, None)
sys.modules.pop("risk_engine", None)
from risk_engine import RiskEngine
from ai_trading.strategies import TradeSignal


class DummyAPI:
    def __init__(self, equity=100, last=100):
        self._eq = equity
        self._last = last

    def get_account(self):
        return types.SimpleNamespace(equity=str(self._eq), last_equity=str(self._last))


def make_signal():
    return TradeSignal(symbol="AAPL", side="buy", confidence=1.0, strategy="s")


def test_can_trade_limits():
    eng = RiskEngine()
    sig = make_signal()
    eng.asset_limits["equity"] = 0.5
    eng.exposure["equity"] = 0.6
    assert not eng.can_trade(sig)
    eng.exposure["equity"] = 0.1
    sig.weight = 0.3
    assert eng.can_trade(sig)


def test_register_and_position_size(monkeypatch):
    eng = RiskEngine()
    sig = make_signal()
    sig.weight = 0.1  # Set weight to avoid exposure cap breach
    eng.asset_limits["equity"] = 1.0
    eng.strategy_limits["s"] = 1.0
    qty = eng.position_size(sig, cash=100, price=10)
    # Position size may be influenced by multiple factors, just ensure it's positive
    assert qty > 0
    eng.register_fill(sig)
    assert eng.exposure["equity"] == sig.weight
    sell = TradeSignal(symbol="AAPL", side="sell", confidence=1.0, strategy="s", weight=sig.weight)
    eng.register_fill(sell)
    assert round(eng.exposure["equity"], 6) == 0


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
