from types import SimpleNamespace

from ai_trading.core import bot_engine


def test_model_feature_names_uses_feature_names_in_when_present() -> None:
    model = SimpleNamespace(feature_names_in_=["macd", "atr", "vwap", "macds", "sma_50", "sma_200"])

    assert bot_engine._model_feature_names(model) == [
        "macd",
        "atr",
        "vwap",
        "macds",
        "sma_50",
        "sma_200",
    ]


def test_model_feature_names_falls_back_to_live_columns() -> None:
    assert bot_engine._model_feature_names(object()) == list(bot_engine.LIVE_ML_FEATURE_COLUMNS)
