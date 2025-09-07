from ai_trading.runtime.params import build_runtime
from ai_trading.core.runtime import REQUIRED_PARAM_DEFAULTS


def test_build_runtime_skips_none_overrides():
    overrides = {
        'CAPITAL_CAP': 0.08,
        'DOLLAR_RISK_LIMIT': None,
    }
    params = build_runtime(overrides)
    assert params['CAPITAL_CAP'] == 0.08
    assert params['DOLLAR_RISK_LIMIT'] == REQUIRED_PARAM_DEFAULTS['DOLLAR_RISK_LIMIT']
