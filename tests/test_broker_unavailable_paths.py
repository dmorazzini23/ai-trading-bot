import logging
from types import SimpleNamespace

import sys
import types

if "numpy" not in sys.modules:  # pragma: no cover - dependency stub for tests
    numpy_stub = types.ModuleType("numpy")
    numpy_stub.ndarray = object
    numpy_stub.array = lambda *args, **kwargs: args[0] if args else None
    numpy_stub.isnan = lambda value: value != value
    numpy_stub.float64 = float
    numpy_stub.int64 = int
    numpy_stub.nan = float("nan")
    numpy_stub.random = types.SimpleNamespace(seed=lambda *_a, **_k: None)
    sys.modules["numpy"] = numpy_stub

if "portalocker" not in sys.modules:  # pragma: no cover - dependency stub for tests
    portalocker_stub = types.ModuleType("portalocker")
    portalocker_stub.LOCK_EX = 1

    def _noop(*_args, **_kwargs):
        return None

    portalocker_stub.lock = _noop
    portalocker_stub.unlock = _noop
    sys.modules["portalocker"] = portalocker_stub

if "bs4" not in sys.modules:  # pragma: no cover - dependency stub for tests
    bs4_stub = types.ModuleType("bs4")

    class _BeautifulSoup:  # pragma: no cover - simple placeholder
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def find_all(self, *_a, **_k):
            return []

    bs4_stub.BeautifulSoup = _BeautifulSoup
    sys.modules["bs4"] = bs4_stub

from ai_trading.core import bot_engine
from ai_trading.core.bot_engine import check_pdt_rule, safe_alpaca_get_account
from ai_trading.logging.emit_once import reset_emit_once_state


def test_safe_account_none():
    # AI-AGENT-REF: ensure None is returned when Alpaca client missing
    ctx = SimpleNamespace(api=None)
    assert safe_alpaca_get_account(ctx) is None


def test_pdt_rule_skips_without_false_fail(monkeypatch, caplog):
    # AI-AGENT-REF: verify PDT check logs skip and not failure
    reset_emit_once_state()
    monkeypatch.setattr(bot_engine, "_has_alpaca_credentials", lambda: False)
    ctx = SimpleNamespace(api=None)
    with caplog.at_level(logging.INFO):
        assert check_pdt_rule(ctx) is False
    msgs = [r.getMessage() for r in caplog.records]
    assert any("PDT_CHECK_SKIPPED" in m for m in msgs)
    assert not any("PDT_CHECK_FAILED" in m for m in msgs)


def test_check_pdt_rule_returns_false_without_credentials(monkeypatch):
    reset_emit_once_state()
    runtime = SimpleNamespace(api=None)

    def fail(*_args, **_kwargs):  # pragma: no cover - should not be called
        raise AssertionError("should not attempt Alpaca initialization when unavailable")

    monkeypatch.setattr(bot_engine, "_has_alpaca_credentials", lambda: False)
    monkeypatch.setattr(bot_engine, "_initialize_alpaca_clients", fail)
    monkeypatch.setattr(bot_engine, "ensure_alpaca_attached", fail)
    monkeypatch.setattr(bot_engine, "safe_alpaca_get_account", fail)

    before = runtime.__dict__.copy()
    assert check_pdt_rule(runtime) is False
    assert runtime.__dict__ == before


def test_check_pdt_rule_handles_missing_api_attribute(monkeypatch):
    reset_emit_once_state()
    runtime = SimpleNamespace()

    def fail(*_args, **_kwargs):  # pragma: no cover - should not be called
        raise AssertionError("should not attempt Alpaca initialization when unavailable")

    monkeypatch.setattr(bot_engine, "_has_alpaca_credentials", lambda: False)
    monkeypatch.setattr(bot_engine, "_initialize_alpaca_clients", fail)
    monkeypatch.setattr(bot_engine, "ensure_alpaca_attached", fail)
    monkeypatch.setattr(bot_engine, "safe_alpaca_get_account", fail)

    before = runtime.__dict__.copy()
    assert check_pdt_rule(runtime) is False
    assert runtime.__dict__ == before


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
    hb = {}
    monkeypatch.setattr(bot_engine, "_log_loop_heartbeat", lambda *a, **k: hb.setdefault("loop", True))
    monkeypatch.setattr(bot_engine, "_send_heartbeat", lambda: hb.setdefault("halt", True))
    monkeypatch.setattr(bot_engine, "ensure_data_fetcher", lambda _rt: True)
    def fail(*a, **k):
        raise AssertionError("PDT should not be called")
    monkeypatch.setattr(bot_engine, "check_pdt_rule", fail)
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
