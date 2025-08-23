import types
from ai_trading.capital_scaling import update_if_present, capital_scale


class DummyRuntime:
    pass


def test_update_if_present_calls_update_then_reads_scale():
    rt = DummyRuntime()
    state = {"updated": False, "scale": 0.42}

    class Scaler:
        def update(self, runtime, equity):
            state["updated"] = True
            # returns None by design in this module; caller should then read current_scale()
        def current_scale(self):
            return state["scale"]

    rt.capital_scaler = Scaler()
    val = update_if_present(rt, equity=1000.0)
    assert state["updated"] is True
    assert abs(val - 0.42) < 1e-12


def test_update_if_present_narrowed_exceptions_return_one():
    rt = DummyRuntime()

    class BadScaler:
        def update(self, runtime, equity):
            raise TypeError("boom")
        def current_scale(self):  # pragma: no cover - not reached
            return 2.0

    rt.capital_scaler = BadScaler()
    val = update_if_present(rt, equity=1.0)
    assert val == 1.0


def test_capital_scale_narrowed_exceptions_return_one():
    rt = DummyRuntime()

    class BadScaler:
        def current_scale(self):
            raise AttributeError("nope")

    rt.capital_scaler = BadScaler()
    val = capital_scale(rt)
    assert val == 1.0

