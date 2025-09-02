import sys
import types

from ai_trading import main as main_module
from ai_trading.core import bot_engine


def test_preflight_initializes_trade_log(tmp_path, monkeypatch):
    """preflight_import_health creates the trade log file on startup."""
    # Stub alpaca client modules required by preflight_import_health
    monkeypatch.setitem(sys.modules, "alpaca", types.ModuleType("alpaca"))
    monkeypatch.setitem(sys.modules, "alpaca.trading", types.ModuleType("alpaca.trading"))
    monkeypatch.setitem(
        sys.modules, "alpaca.trading.client", types.ModuleType("alpaca.trading.client")
    )

    log_path = tmp_path / "trades.csv"
    monkeypatch.setattr(bot_engine, "TRADE_LOG_FILE", str(log_path), raising=False)
    bot_engine._TRADE_LOGGER_SINGLETON = None

    main_module.preflight_import_health()

    assert log_path.exists()
    assert log_path.stat().st_size > 0
