import logging
from types import SimpleNamespace

import sys
import types
from typing import Any, cast

if "numpy" not in sys.modules:  # pragma: no cover - dependency stub for tests
    numpy_stub = cast(Any, types.ModuleType("numpy"))
    numpy_stub.ndarray = object
    numpy_stub.array = lambda *args, **kwargs: args[0] if args else None
    numpy_stub.isnan = lambda value: value != value
    numpy_stub.float64 = float
    numpy_stub.int64 = int
    numpy_stub.nan = float("nan")
    numpy_stub.random = types.SimpleNamespace(seed=lambda *_a, **_k: None)
    sys.modules["numpy"] = numpy_stub

if "portalocker" not in sys.modules:  # pragma: no cover - dependency stub for tests
    portalocker_stub = cast(Any, types.ModuleType("portalocker"))
    portalocker_stub.LOCK_EX = 1

    def _noop(*_args, **_kwargs):
        return None

    portalocker_stub.lock = _noop
    portalocker_stub.unlock = _noop
    sys.modules["portalocker"] = portalocker_stub

if "bs4" not in sys.modules:  # pragma: no cover - dependency stub for tests
    bs4_stub = cast(Any, types.ModuleType("bs4"))

    class _BeautifulSoup:  # pragma: no cover - simple placeholder
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def find_all(self, *_a, **_k):
            return []

    bs4_stub.BeautifulSoup = _BeautifulSoup
    sys.modules["bs4"] = bs4_stub

from ai_trading.core import bot_engine
from ai_trading.core.bot_engine import safe_alpaca_get_account
from ai_trading.logging.emit_once import reset_emit_once_state


def test_safe_account_none():
    # AI-AGENT-REF: ensure None is returned when Alpaca client missing
    ctx = SimpleNamespace(api=None)
    assert safe_alpaca_get_account(ctx) is None


def test_run_all_trades_aborts_without_api(monkeypatch, caplog):
    """run_all_trades_worker should abort early when Alpaca client missing."""
    reset_emit_once_state()
    state = bot_engine.BotState()
    monkeypatch.setenv("SHADOW_MODE", "true")
    monkeypatch.setenv("WEBHOOK_SECRET", "test")
    runtime = bot_engine.get_ctx()
    runtime.api = None
    monkeypatch.setattr(bot_engine, "is_market_open", lambda: True)
    monkeypatch.setattr(bot_engine, "_ensure_alpaca_classes", lambda: None)
    monkeypatch.setattr(bot_engine, "_ALPACA_IMPORT_ERROR", None)
    monkeypatch.setattr(bot_engine, "get_minute_df", lambda *a, **k: None)
    monkeypatch.setattr(bot_engine, "last_minute_bar_age_seconds", lambda *a, **k: 0)
    monkeypatch.setattr(bot_engine, "get_cached_minute_timestamp", lambda *a, **k: 0)
    hb: dict[str, bool] = {}
    monkeypatch.setattr(bot_engine, "_log_loop_heartbeat", lambda *a, **k: hb.setdefault("loop", True))
    monkeypatch.setattr(bot_engine, "_send_heartbeat", lambda: hb.setdefault("halt", True))
    monkeypatch.setattr(bot_engine, "ensure_data_fetcher", lambda _rt: True)
    with caplog.at_level(logging.WARNING):
        bot_engine.run_all_trades_worker(state, runtime)
    assert hb.get("loop") and not hb.get("halt")
    msgs = [r.getMessage() for r in caplog.records]
    assert any("ALPACA_CLIENT_MISSING" in m for m in msgs)
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    errors = [
        r
        for r in caplog.records
        if r.levelno >= logging.ERROR and "Market status check failed" not in r.getMessage()
    ]
    assert not errors
