from capital_scaling import drawdown_adjusted_kelly


def test_drawdown_tapering():
    peak = 100000
    value = 90000
    k = 0.1
    adj = drawdown_adjusted_kelly(value, peak, k)
    assert adj < k
    value = 50000
    adj2 = drawdown_adjusted_kelly(value, peak, k)
    assert adj2 < adj
    assert adj2 > 0
