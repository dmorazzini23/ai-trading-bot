def test_mean_reversion_strategy_public_import():
    from ai_trading.strategies import MeanReversionStrategy

    assert hasattr(MeanReversionStrategy, "__call__") or hasattr(
        MeanReversionStrategy, "__init__"
    )

