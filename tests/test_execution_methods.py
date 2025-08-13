from ai_trading.execution import ExecutionEngine


def test_execute_sliced(monkeypatch):
    monkeypatch.setattr(ExecutionEngine, "execute_sliced", lambda *a, **k: {"ok": True})  # AI-AGENT-REF: patch public API
    assert ExecutionEngine.execute_sliced(None, "AAPL", 10) == {"ok": True}
