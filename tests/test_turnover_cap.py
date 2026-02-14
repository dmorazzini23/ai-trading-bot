from ai_trading.core.netting import apply_turnover_gate


def test_turnover_cap_blocks():
    assert apply_turnover_gate(1200.0, 1000.0)
    assert not apply_turnover_gate(500.0, 1000.0)
