from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

import ai_trading.signals as signals_mod
from ai_trading.signals import ensemble_vote_signals, generate_ensemble_signal, generate_signal
from ai_trading.strategies import metalearning as ml_mod
from ai_trading.strategies.base import StrategySignal
from ai_trading.strategies.metalearning import MetaLearning


def _price_frame(rows: int = 80, start: float = 100.0, step: float = 0.05) -> pd.DataFrame:
    index = pd.date_range("2026-01-01 14:30", periods=rows, freq="min", tz="UTC")
    close = pd.Series(start + np.arange(rows) * step, index=index)
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(rows, 1_000),
        },
        index=index,
    )


def _feature_row(**overrides: float) -> pd.DataFrame:
    values = {
        "rsi": 50.0,
        "macd_bullish": 0.0,
        "macd_bearish": 0.0,
        "bb_breakout_up": 0.0,
        "bb_breakout_down": 0.0,
        "sma_cross_bullish": 0.0,
        "sma_cross_bearish": 0.0,
        "momentum_5": 0.0,
    }
    values.update(overrides)
    return pd.DataFrame([values])


def test_generate_signal_maps_nonfinite_values_and_rejects_invalid_inputs() -> None:
    index = pd.date_range("2026-01-02", periods=6, freq="min", tz="UTC")
    df = pd.DataFrame({"momentum": [-2.5, 0.0, 3.5, np.nan, np.inf, -np.inf]}, index=index)

    signal = generate_signal(df, "momentum")

    assert signal.tolist() == [-1, 0, 1, 0, 0, 0]
    assert signal.index.equals(index)
    assert signal.name == "signal"

    with pytest.raises(ValueError, match="df cannot be None"):
        generate_signal(None, "momentum")
    with pytest.raises(TypeError, match="df must be a pandas DataFrame"):
        generate_signal([1, 2, 3], "momentum")
    with pytest.raises(ValueError, match="df cannot be empty"):
        generate_signal(pd.DataFrame(columns=["momentum"]), "momentum")
    with pytest.raises(TypeError, match="column must be a string"):
        generate_signal(df, 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="column 'missing' not found"):
        generate_signal(df, "missing")


def test_score_candidates_clips_model_outputs_and_generate_signals_copies() -> None:
    features = pd.DataFrame({"alpha": [0.1, 0.2, 0.3]})

    class ProbaModel:
        def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
            assert data is features
            return np.array([[0.2, 1.4], [0.9, -0.5], [0.6, 0.35]])

    scored = signals_mod.score_candidates(features, ProbaModel())

    assert scored["score"].tolist() == [1.0, 0.0, 0.35]
    assert "score" not in features

    generated = signals_mod.generate_signals(scored, buy_threshold=0.75)

    assert generated.equals(scored)
    assert generated is not scored


def test_compute_signal_matrix_does_not_backfill_rsi_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = pd.date_range("2026-01-02 14:30", periods=30, freq="min", tz="UTC")
    close = pd.Series(np.linspace(100.0, 103.0, len(index)), index=index)
    close.iloc[0] = np.nan
    df = pd.DataFrame(
        {
            "close": close,
            "high": close.ffill().fillna(100.0) + 1.0,
            "low": close.ffill().fillna(100.0) - 1.0,
        },
        index=index,
    )
    captured: dict[str, tuple[float, ...]] = {}

    def fake_rsi(prices: tuple[float, ...], period: int) -> pd.Series:
        captured["prices"] = prices
        assert period == 14
        return pd.Series([50.0] * len(prices), index=index)

    monkeypatch.setattr(signals_mod, "rsi", fake_rsi)

    matrix = signals_mod.compute_signal_matrix(df)

    assert matrix is not None
    assert np.isnan(captured["prices"][0])
    assert captured["prices"][1] == pytest.approx(close.iloc[1])


def test_signal_aggregation_votes_and_breakout_branches() -> None:
    index = pd.date_range("2026-01-03", periods=3, freq="min", tz="UTC")
    matrix = pd.DataFrame(
        {
            "macd": [0.7, -0.8, 0.6],
            "rsi": [0.9, -0.2, 0.0],
            "sma_diff": [0.1, -0.9, 0.0],
        },
        index=index,
    )

    votes = ensemble_vote_signals(matrix)

    assert votes.tolist() == [1, -1, 0]
    assert ensemble_vote_signals(pd.DataFrame()).empty
    assert signals_mod.classify_regime(pd.DataFrame()).empty

    sell_breakout = pd.DataFrame({"close": [111.0], "UB": [110.0]})
    lower_band_buy = pd.DataFrame({"close": [89.0], "LB": [90.0]})
    neutral = pd.DataFrame({"close": [100.0], "UB": [110.0], "LB": [90.0]})

    assert generate_ensemble_signal(sell_breakout) == -1
    assert generate_ensemble_signal(lower_band_buy) == 1
    assert generate_ensemble_signal(neutral) == 0


def test_execute_strategy_handles_invalid_short_training_and_prediction_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = MetaLearning()

    assert strategy.execute_strategy()["signal"] == "hold"
    assert strategy.execute_strategy(_price_frame(rows=10), symbol="AAPL")["signal"] == "hold"

    enough_data = _price_frame(rows=60)
    monkeypatch.setattr(strategy, "train_model", lambda data: False)
    assert strategy.execute_strategy(enough_data, symbol="AAPL")["signal"] == "hold"

    strategy = MetaLearning()
    strategy.is_trained = True
    strategy.last_training_date = datetime.now(UTC)
    monkeypatch.setattr(strategy, "predict_price_movement", lambda data: None)
    assert strategy.execute_strategy(enough_data, symbol="AAPL")["signal"] == "hold"

    cached = {"signal": "buy", "confidence": 0.7, "strength": 0.6}
    strategy._cache_prediction("MSFT", cached)
    result = strategy.execute_strategy("MSFT")

    assert result == cached
    assert result is not strategy.prediction_cache["MSFT"]


def test_metalearning_generate_signals_aggregates_non_hold_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = MetaLearning()
    strategy.symbols = ["AAPL", "MSFT", "SPY"]
    strategy.prediction_accuracy = 0.72
    bar_ts = datetime(2026, 1, 5, 15, 45, tzinfo=UTC)
    predictions = {
        "AAPL": {
            "signal": "buy",
            "confidence": 0.8,
            "strength": 0.7,
            "price_target": 111.0,
            "stop_loss": 98.0,
            "reasoning": "bullish branch",
            "bar_ts": bar_ts,
        },
        "MSFT": {
            "signal": "sell",
            "confidence": 0.75,
            "strength": 0.65,
            "price_target": 95.0,
            "stop_loss": 104.0,
            "reasoning": "bearish branch",
        },
        "SPY": {"signal": "hold"},
    }

    def fake_execute(data: object = None, symbol: str | None = None) -> dict[str, object]:
        selected = symbol or data
        return predictions[str(selected)]

    monkeypatch.setattr(strategy, "execute_strategy", fake_execute)

    signals = strategy.generate_signals({})

    assert [signal.symbol for signal in signals] == ["AAPL", "MSFT"]
    assert [signal.side for signal in signals] == ["buy", "sell"]
    assert signals[0].metadata["bar_ts"] == bar_ts.isoformat()
    assert signals[1].metadata["model_accuracy"] == 0.72
    assert strategy.signals_generated == 2


def test_metalearning_generate_signals_uses_supplied_backtest_frames_without_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = MetaLearning()
    strategy.symbols = ["AAPL"]
    strategy.is_trained = True
    strategy.last_training_date = datetime.now(UTC)
    strategy.prediction_accuracy = 0.8
    frame = _price_frame(rows=80)

    def blocked_fetch(*_args: object, **_kwargs: object) -> pd.DataFrame:
        raise AssertionError("generate_signals fetched fresh data")

    def fake_predict(data: pd.DataFrame) -> dict[str, object]:
        assert data is frame
        return {
            "direction": "buy",
            "confidence": 0.9,
            "current_price": float(data["close"].iloc[-1]),
            "volatility": 0.02,
            "probability_distribution": {"buy": 0.9, "sell": 0.05, "hold": 0.05},
        }

    strategy.prediction_cache["AAPL"] = {"signal": "sell", "confidence": 0.9, "strength": 0.9}
    strategy.cache_expiry["AAPL"] = datetime.now(UTC) + timedelta(hours=1)
    monkeypatch.setattr(ml_mod, "get_minute_df", blocked_fetch)
    monkeypatch.setattr(strategy, "predict_price_movement", fake_predict)

    signals = strategy.generate_signals({"frames": {"AAPL": frame}, "backtest": True})

    assert [(signal.symbol, signal.side) for signal in signals] == [("AAPL", "buy")]
    assert strategy.prediction_cache["AAPL"]["signal"] == "sell"


def test_metalearning_generate_signals_skips_missing_backtest_symbol_without_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = MetaLearning()
    strategy.symbols = ["AAPL"]

    def blocked_fetch(*_args: object, **_kwargs: object) -> pd.DataFrame:
        raise AssertionError("missing backtest data fetched fresh data")

    monkeypatch.setattr(ml_mod, "get_minute_df", blocked_fetch)

    assert strategy.generate_signals({"frames": {}, "mode": "backtest"}) == []


def test_train_model_exception_failure_does_not_mark_trained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StandardScaler:
        pass

    Preprocessing = type("Preprocessing", (), {"StandardScaler": StandardScaler})
    Ensemble = type(
        "Ensemble",
        (),
        {"RandomForestClassifier": object, "GradientBoostingClassifier": object},
    )

    class Metrics:
        @staticmethod
        def accuracy_score(*_args: object, **_kwargs: object) -> float:
            return 0.0

    strategy = MetaLearning()
    monkeypatch.setattr(ml_mod, "ML_AVAILABLE", True)
    monkeypatch.setattr(ml_mod, "PANDAS_AVAILABLE", True)
    monkeypatch.setattr(ml_mod, "load_sklearn_preprocessing", lambda: Preprocessing)
    monkeypatch.setattr(ml_mod, "load_sklearn_ensemble", lambda: Ensemble)
    monkeypatch.setattr(ml_mod, "load_sklearn_metrics", lambda: Metrics)
    monkeypatch.setattr(
        strategy,
        "extract_features",
        lambda _data: (_ for _ in ()).throw(ValueError("feature failure")),
    )

    assert strategy.train_model(_price_frame(rows=80)) is False
    assert strategy.is_trained is False
    assert strategy.last_training_date is None


def test_calculate_position_size_applies_confidence_strength_and_accuracy() -> None:
    strategy = MetaLearning()
    strategy.prediction_accuracy = 0.5
    signal = StrategySignal("AAPL", "buy", strength=0.8, confidence=0.75)

    assert strategy.calculate_position_size(signal, portfolio_value=10_000) == 150


@pytest.mark.parametrize(
    ("features", "direction"),
    [
        (
            _feature_row(
                rsi=20.0,
                macd_bullish=1.0,
                bb_breakout_up=1.0,
                sma_cross_bullish=1.0,
                momentum_5=0.05,
            ),
            "buy",
        ),
        (
            _feature_row(
                rsi=85.0,
                macd_bearish=1.0,
                bb_breakout_down=1.0,
                sma_cross_bearish=1.0,
                momentum_5=-0.05,
            ),
            "sell",
        ),
        (_feature_row(), "hold"),
    ],
)
def test_missing_model_uses_technical_fallback_with_bounded_confidence(
    monkeypatch: pytest.MonkeyPatch,
    features: pd.DataFrame,
    direction: str,
) -> None:
    strategy = MetaLearning()
    strategy.is_trained = True
    monkeypatch.setattr(ml_mod, "PANDAS_AVAILABLE", True)
    monkeypatch.setattr(strategy, "extract_features", lambda data: features)

    result = strategy.predict_price_movement(_price_frame())

    assert result is not None
    assert result["source"] == "technical_fallback"
    assert result["direction"] == direction
    assert 0.0 <= result["confidence"] <= 0.8
    assert all(0.0 <= value <= 1.0 for value in result["probability_distribution"].values())


def test_ml_prediction_clips_confidence_and_characterizes_two_and_three_class_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class IdentityScaler:
        def transform(self, features: pd.DataFrame) -> pd.DataFrame:
            return features

    class Model:
        def __init__(self, probabilities: list[float]) -> None:
            self.probabilities = np.array(probabilities)

        def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
            assert not features.empty
            return np.array([self.probabilities])

    strategy = MetaLearning()
    strategy.is_trained = True
    strategy.feature_columns = ["f1"]
    strategy.scaler = IdentityScaler()
    strategy.rf_model = Model([-0.5, 1.6])
    strategy.gb_model = Model([0.1, 1.2])
    monkeypatch.setattr(ml_mod, "ML_AVAILABLE", True)
    monkeypatch.setattr(ml_mod, "PANDAS_AVAILABLE", True)
    monkeypatch.setattr(strategy, "extract_features", lambda data: pd.DataFrame({"f1": [1.0]}))

    two_class = strategy.predict_price_movement(_price_frame())

    assert two_class is not None
    assert two_class["direction"] == "buy"
    assert two_class["confidence"] == 1.0
    assert two_class["probability_distribution"]["buy"] == 1.0

    strategy.rf_model = Model([0.05, 0.75, 0.2])
    strategy.gb_model = Model([0.05, 0.85, 0.1])

    three_class = strategy.predict_price_movement(_price_frame())

    assert three_class is not None
    assert three_class["direction"] == "hold"
    assert three_class["probability_distribution"]["hold"] == pytest.approx(0.79)
    assert 0.0 <= three_class["confidence"] <= 1.0


def test_model_prediction_failure_falls_back_to_technical_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class IdentityScaler:
        def transform(self, features: pd.DataFrame) -> pd.DataFrame:
            return features

    class RaisingModel:
        def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
            raise ValueError("model unavailable")

    strategy = MetaLearning()
    strategy.is_trained = True
    strategy.feature_columns = ["f1"]
    strategy.scaler = IdentityScaler()
    strategy.rf_model = RaisingModel()
    strategy.gb_model = RaisingModel()
    monkeypatch.setattr(ml_mod, "ML_AVAILABLE", True)
    monkeypatch.setattr(strategy, "extract_features", lambda data: _feature_row(momentum_5=0.05, f1=1.0))

    result = strategy.predict_price_movement(_price_frame())

    assert result is not None
    assert result["source"] == "technical_fallback"
    assert result["direction"] == "buy"


def test_convert_prediction_to_signal_bounds_low_confidence_and_error_paths() -> None:
    strategy = MetaLearning()
    strategy.prediction_accuracy = 0.8

    buy = strategy._convert_prediction_to_signal(
        "AAPL",
        {
            "direction": "buy",
            "confidence": 0.9,
            "current_price": 100.0,
            "volatility": 0.03,
            "probability_distribution": {"buy": 0.9, "sell": 0.05, "hold": 0.05},
        },
        _price_frame(),
    )
    sell = strategy._convert_prediction_to_signal(
        "MSFT",
        {
            "direction": "sell",
            "confidence": 0.7,
            "current_price": 100.0,
            "volatility": 0.02,
            "probability_distribution": {"buy": 0.1, "sell": 0.7, "hold": 0.2},
        },
        _price_frame(),
    )
    low_confidence = strategy._convert_prediction_to_signal(
        "SPY",
        {
            "direction": "buy",
            "confidence": 0.1,
            "current_price": 100.0,
            "volatility": 0.02,
            "probability_distribution": {"buy": 0.6, "sell": 0.2, "hold": 0.2},
        },
        _price_frame(),
    )
    malformed = strategy._convert_prediction_to_signal(
        "QQQ",
        {"direction": "buy", "confidence": 0.9, "current_price": 100.0, "volatility": 0.02},
        _price_frame(),
    )

    assert buy["signal"] == "buy"
    assert buy["strength"] == pytest.approx(0.72)
    assert buy["price_target"] == pytest.approx(106.0)
    assert buy["stop_loss"] == pytest.approx(97.0)
    assert sell["signal"] == "sell"
    assert sell["price_target"] == pytest.approx(96.0)
    assert sell["stop_loss"] == pytest.approx(102.0)
    assert low_confidence["signal"] == "hold"
    assert malformed["signal"] == "hold"


def test_retrain_and_cache_branch_characterization() -> None:
    strategy = MetaLearning()

    assert strategy._should_retrain()
    assert not strategy._is_cached_prediction_valid("AAPL")

    strategy.is_trained = True
    strategy.last_training_date = datetime.now(UTC)
    assert not strategy._should_retrain()

    strategy.last_training_date = datetime.now(UTC) - timedelta(days=8)
    assert strategy._should_retrain()

    strategy.prediction_cache["AAPL"] = {"signal": "buy"}
    assert not strategy._is_cached_prediction_valid("AAPL")

    strategy.cache_expiry["AAPL"] = datetime.now(UTC) - timedelta(seconds=1)
    assert not strategy._is_cached_prediction_valid("AAPL")
