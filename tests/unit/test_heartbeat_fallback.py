import logging
import types
from unittest.mock import Mock

import ai_trading.core.bot_engine as eng


def test_heartbeat_persists_with_fallback(monkeypatch, caplog):
    """Ensure heartbeat uses fallback data source when Alpaca API missing."""

    class DummyRiskEngine:
        def wait_for_exposure_update(self, timeout: float) -> None:
            pass

    state = eng.BotState()
    runtime = types.SimpleNamespace(api=None, risk_engine=DummyRiskEngine())

    monkeypatch.setattr(eng, "_ensure_alpaca_classes", lambda: None)
    monkeypatch.setattr(eng, "_init_metrics", lambda: None)
    monkeypatch.setattr(eng, "is_market_open", lambda: True)
    monkeypatch.setattr(eng, "ensure_alpaca_attached", lambda _rt: None)
    monkeypatch.setattr(eng, "_validate_trading_api", lambda _api: False)

    fetch_called = Mock()
    monkeypatch.setattr(eng, "ensure_data_fetcher", lambda _rt: fetch_called())

    heartbeat = {"log": False, "halt": False}
    monkeypatch.setattr(eng, "_log_loop_heartbeat", lambda *a, **k: heartbeat.__setitem__("log", True))
    monkeypatch.setattr(eng, "_send_heartbeat", lambda: heartbeat.__setitem__("halt", True))

    class DummyLock:
        def acquire(self, blocking: bool = False) -> bool:
            return True

        def release(self) -> None:
            pass

    monkeypatch.setattr(eng, "run_lock", DummyLock())

    with caplog.at_level(logging.WARNING):
        eng.run_all_trades_worker(state, runtime)

    assert fetch_called.called
    assert heartbeat["log"]
    assert not heartbeat["halt"]
    assert "ALPACA_CLIENT_MISSING_FALLBACK_ACTIVE" in caplog.text
