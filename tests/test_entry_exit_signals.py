import pytest
pd = pytest.importorskip("pandas")
from types import SimpleNamespace
import sys
import types

# Stub heavy dependencies before importing trade_logic
logger_mod = types.ModuleType("ai_trading.logging")
logger_mod.get_logger = lambda name: types.SimpleNamespace(
    info=lambda *a, **k: None,
    debug=lambda *a, **k: None,
    warning=lambda *a, **k: None,
)
sys.modules.setdefault("ai_trading.logging", logger_mod)

cap_mod = types.ModuleType("ai_trading.capital_scaling")
cap_mod.drawdown_adjusted_kelly = lambda *a, **k: 1.0
cap_mod.drawdown_adjusted_kelly_alt = cap_mod.drawdown_adjusted_kelly
sys.modules.setdefault("ai_trading.capital_scaling", cap_mod)

settings_mod = types.ModuleType("ai_trading.config.settings")
settings_mod.get_settings = lambda: SimpleNamespace(intraday_lookback_minutes=120)
sys.modules.setdefault("ai_trading.config.settings", settings_mod)

core_mod = types.ModuleType("ai_trading.core.bot_engine")
core_mod._fetch_intraday_bars_chunked = lambda *a, **k: {}
sys.modules.setdefault("ai_trading.core.bot_engine", core_mod)

from ai_trading.trade_logic import _compute_entry_signal, _compute_exit_signal


def test_entry_signal_buy_on_cross():
    df = pd.DataFrame({"close": [3, 2, 2, 3]})
    ctx = SimpleNamespace(entry_short_window=2, entry_long_window=3)
    sig = _compute_entry_signal(ctx, "SYM", df)
    assert sig == {"buy": True, "price": 3.0}


def test_exit_signal_sell_on_cross():
    df = pd.DataFrame({"close": [2, 3, 3, 2]})
    ctx = SimpleNamespace(exit_short_window=2, exit_long_window=3)
    sig = _compute_exit_signal(ctx, "SYM", df)
    assert sig == {"sell": True, "price": 2.0}
