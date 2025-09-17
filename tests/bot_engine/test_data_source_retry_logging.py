from __future__ import annotations

import sys
import types

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
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    bs4_stub.BeautifulSoup = _BeautifulSoup
    sys.modules["bs4"] = bs4_stub

if "flask" not in sys.modules:  # pragma: no cover - optional dependency shim
    flask_stub = types.ModuleType("flask")

    class _Flask:  # pragma: no cover - minimal placeholder
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def route(self, *_a, **_k):
            def decorator(func):
                return func

            return decorator

    flask_stub.Flask = _Flask
    flask_stub.jsonify = lambda *a, **k: None
    flask_stub.Response = object
    sys.modules["flask"] = flask_stub

import pytest

from ai_trading.core import bot_engine as bot


class DummyRiskEngine:
    def wait_for_exposure_update(self, timeout: float) -> None:
        return None

    def refresh_positions(self, api) -> None:
        return None

    def _adaptive_global_cap(self) -> float:
        return 0.0


class DummyAPI:
    def list_positions(self) -> list:
        return []

    def get_account(self):
        return types.SimpleNamespace(cash=1000.0, equity=1000.0, last_equity=900.0)


class DummyLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_data_source_retry_marks_failure(monkeypatch, caplog):
    state = bot.BotState()
    runtime = types.SimpleNamespace(
        risk_engine=DummyRiskEngine(),
        api=DummyAPI(),
        execution_engine=None,
        data_fetcher=types.SimpleNamespace(_minute_timestamps={}),
        model=object(),
        tickers=["AAA", "BBB"],
        portfolio_weights={},
    )

    setattr(bot.CFG, "log_market_fetch", False)
    setattr(bot.CFG, "shadow_mode", False)

    dummy_lock = DummyLock()
    monkeypatch.setattr(bot, "portfolio_lock", dummy_lock, raising=False)
    import ai_trading.utils as utils_mod

    monkeypatch.setattr(utils_mod, "portfolio_lock", dummy_lock, raising=False)
    monkeypatch.setattr(bot.portfolio, "compute_portfolio_weights", lambda runtime, symbols: {})

    monkeypatch.setattr(bot, "_ALPACA_IMPORT_ERROR", None, raising=False)
    monkeypatch.setattr(bot, "_ensure_alpaca_classes", lambda: None)
    monkeypatch.setattr(bot, "_init_metrics", lambda: None)
    monkeypatch.setattr(bot, "_ensure_execution_engine", lambda runtime: None)
    monkeypatch.setattr(bot, "ensure_alpaca_attached", lambda runtime: None)
    monkeypatch.setattr(bot, "_validate_trading_api", lambda api: True)
    monkeypatch.setattr(bot, "check_pdt_rule", lambda runtime: False)
    monkeypatch.setattr(bot, "get_trade_cooldown_min", lambda: 0)
    monkeypatch.setattr(bot, "is_market_open", lambda: True)
    monkeypatch.setattr(bot, "get_verbose_logging", lambda: False)
    monkeypatch.setattr(bot, "list_open_orders", lambda api: [])
    monkeypatch.setattr(bot, "ensure_data_fetcher", lambda runtime: None)
    monkeypatch.setattr(bot, "get_trade_logger", lambda: None)
    monkeypatch.setattr(bot, "_get_runtime_context_or_none", lambda: None)
    monkeypatch.setattr(
        bot,
        "_prepare_run",
        lambda runtime, state, tickers: (1000.0, True, ["AAA", "BBB"]),
    )
    monkeypatch.setattr(bot, "run_multi_strategy", lambda runtime: None)
    monkeypatch.setattr(bot, "_send_heartbeat", lambda: None)
    monkeypatch.setattr(bot, "_log_loop_heartbeat", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot, "_check_runtime_stops", lambda runtime: None)
    monkeypatch.setattr(bot, "check_halt_flag", lambda runtime: False)
    monkeypatch.setattr(bot, "manage_position_risk", lambda runtime, pos: None)
    monkeypatch.setattr(bot.time, "sleep", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot, "MEMORY_OPTIMIZATION_AVAILABLE", False, raising=False)
    monkeypatch.setattr(bot, "get_strategies", lambda: [])
    monkeypatch.setattr(
        bot,
        "_process_symbols",
        lambda symbols, current_cash, alpha_model, regime_ok: (
            ["AAA"],
            {"AAA": 5, "BBB": 0},
        ),
    )

    caplog.set_level("INFO")

    bot.run_all_trades_worker(state, runtime)

    records = [
        record for record in caplog.records if record.getMessage() == "DATA_SOURCE_RETRY_FINAL"
    ]
    assert records, "Expected DATA_SOURCE_RETRY_FINAL log entry"
    assert records[-1].success is False
