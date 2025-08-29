from types import SimpleNamespace
from ai_trading.core.bot_engine import _check_runtime_stops


class DummyEngine:
    def __init__(self):
        self.called = False

    def check_stops(self):
        self.called = True


def test_check_stops_invoked_when_present():
    runtime = SimpleNamespace(exec_engine=DummyEngine())
    _check_runtime_stops(runtime)
    assert runtime.exec_engine.called


def test_warning_when_exec_engine_missing(caplog):
    runtime = SimpleNamespace()
    with caplog.at_level("WARNING"):
        _check_runtime_stops(runtime)
    assert "risk-stop checks skipped" in caplog.text
