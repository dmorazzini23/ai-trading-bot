import types

from ai_trading.capital_scaling import update_if_present


def test_update_returns_one_when_absent():
    runtime = types.SimpleNamespace(capital_scaler=None)
    assert update_if_present(runtime, equity=1000) == 1.0


class _Scaler:
    def __init__(self, value: float):
        self.value = value
        self.args = None

    def update(self, runtime, equity):
        self.args = (runtime, equity)
        return self.value


def test_update_forwards_when_present():
    scaler = _Scaler(0.5)
    runtime = types.SimpleNamespace(capital_scaler=scaler)
    assert update_if_present(runtime, equity=2000) == 0.5
    assert scaler.args == (runtime, 2000)
