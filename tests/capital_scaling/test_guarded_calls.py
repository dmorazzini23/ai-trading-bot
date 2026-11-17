import types
from ai_trading.capital_scaling import capital_scale




def test_scale_defaults_to_one_when_absent():
    runtime = types.SimpleNamespace(capital_scaler=None)
    assert capital_scale(runtime) == 1.0
