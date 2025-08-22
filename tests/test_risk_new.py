import pandas as pd

import ai_trading.risk.engine as risk_engine  # AI-AGENT-REF: normalized import


def test_stop_levels():
    stop, take = risk_engine.compute_stop_levels(100, 2)
    assert stop == 98 and take == 104


def test_corr_weights():
    corr = pd.DataFrame([[1,0.5],[0.5,1]], columns=["A","B"], index=["A","B"])
    base = {"A":1.0, "B":1.0}
    res = risk_engine.correlation_position_weights(corr, base)
    assert all(res[k] <= 1.0 for k in res)


def test_drawdown_circuit():
    assert risk_engine.drawdown_circuit([-0.3])
    assert not risk_engine.drawdown_circuit([-0.1])


def test_volatility_filter():
    assert risk_engine.volatility_filter(1, 100)
