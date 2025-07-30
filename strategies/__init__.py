try:
    # Try to import all strategies - this may fail if dependencies aren't available
    from strategies.base import Strategy, TradeSignal, asset_class_for
    from strategies.mean_reversion import MeanReversionStrategy
    from strategies.momentum import MomentumStrategy
    from strategies.moving_average_crossover import MovingAverageCrossoverStrategy

    __all__ = (
        "Strategy",
        "TradeSignal",
        "MomentumStrategy",
        "MeanReversionStrategy",
        "MovingAverageCrossoverStrategy",
        "asset_class_for",
    )
except ImportError:
    # Fallback: only import the base classes if other imports fail (missing dependencies)
    from strategies.base import Strategy, TradeSignal, asset_class_for
    
    __all__ = (
        "Strategy",
        "TradeSignal",
        "asset_class_for",
    )
