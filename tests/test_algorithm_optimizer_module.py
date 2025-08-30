from ai_trading.algorithm_optimizer import (
    AlgorithmOptimizer,
    MarketConditions,
    MarketRegime,
    TradingPhase,
)


def _conditions() -> MarketConditions:
    return MarketConditions(
        regime=MarketRegime.SIDEWAYS,
        volatility=0.2,
        trend_strength=0.5,
        volume_profile=1.0,
        correlation_to_market=0.5,
        sector_rotation=0.0,
        vix_level=20.0,
        time_of_day=TradingPhase.MID_DAY,
    )


def test_position_and_risk_calculations():
    opt = AlgorithmOptimizer()
    cond = _conditions()
    size = opt.calculate_optimal_position_size("T", 10.0, 10000.0, 0.2, cond)
    assert size > 0
    stop = opt.calculate_stop_loss(10.0, "BUY", 0.2, 0.5)
    take = opt.calculate_take_profit(10.0, "BUY", stop)
    assert stop < 10.0 < take
    params = opt.optimize_parameters(cond, [0.01, -0.02, 0.03], force_optimization=True)
    assert params.position_size_multiplier != opt.base_parameters.position_size_multiplier

