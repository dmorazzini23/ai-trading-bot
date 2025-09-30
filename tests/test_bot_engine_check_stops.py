import importlib
import sys
from types import SimpleNamespace
from types import ModuleType

def _import_bot_engine():
    return importlib.import_module("ai_trading.core.bot_engine")


class DummyEngine:
    def __init__(self):
        self.called = False

    def check_stops(self):
        self.called = True


def test_check_stops_invoked_when_present():
    bot_engine = _import_bot_engine()
    runtime = SimpleNamespace(exec_engine=DummyEngine())
    bot_engine._check_runtime_stops(runtime)
    assert runtime.exec_engine.called


def test_warning_when_exec_engine_missing(caplog):
    bot_engine = _import_bot_engine()
    runtime = SimpleNamespace()
    with caplog.at_level("WARNING"):
        bot_engine._check_runtime_stops(runtime)
    assert "risk-stop checks skipped" in caplog.text


def test_stub_execution_engine_avoids_warning(monkeypatch, caplog):
    fake_exec_pkg = ModuleType("ai_trading.execution")
    monkeypatch.setitem(sys.modules, "ai_trading.execution", fake_exec_pkg)
    monkeypatch.delitem(sys.modules, "ai_trading.execution.engine", raising=False)
    monkeypatch.delitem(sys.modules, "ai_trading.core.bot_engine", raising=False)

    stub_bot_engine = importlib.import_module("ai_trading.core.bot_engine")
    runtime = SimpleNamespace(exec_engine=stub_bot_engine.ExecutionEngine())

    with caplog.at_level("WARNING"):
        stub_bot_engine._check_runtime_stops(runtime)

    assert "risk-stop checks skipped" not in caplog.text

    monkeypatch.delitem(sys.modules, "ai_trading.core.bot_engine", raising=False)
    _import_bot_engine()
