from ai_trading.risk.parameters import optimize_stop_loss_multiplier


def test_optimize_stop_loss_multiplier_returns_expected_value():
    """Optimization should apply proportional reduction and yield 1.8."""
    assert optimize_stop_loss_multiplier(2.0, 0.1) == 1.8
