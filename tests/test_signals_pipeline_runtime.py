from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
from pandas import DataFrame

from ai_trading import signals


def _market_frame(rows: int = 60) -> DataFrame:
    close = pd.Series(np.linspace(100.0, 112.0, rows))
    return pd.DataFrame(
        {
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.linspace(1_000, 2_000, rows),
            "EMA_5": close + 1.0,
            "EMA_20": close - 1.0,
            "RSI": [45.0] * rows,
            "SMA_50": close - 2.0,
            "SMA_200": close - 3.0,
            "UB": close + 10.0,
            "LB": close - 10.0,
        }
    )


def test_score_candidates_uses_predict_proba_predict_and_errors():
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

    class ProbaModel:
        def predict_proba(self, _x):
            return np.array([[0.8, 0.2], [0.1, 0.9]])

    class OneDimModel:
        def predict_proba(self, _x):
            return np.array([0.25, 0.75])

    class PredictModel:
        def predict(self, _x):
            return [2.0, -1.0]

    assert list(signals.score_candidates(frame, ProbaModel())["score"]) == [0.2, 0.9]
    assert list(signals.score_candidates(frame, OneDimModel())["score"]) == [0.25, 0.75]
    assert list(signals.score_candidates(frame, PredictModel())["score"]) == [1.0, 0.0]
    with pytest.raises(AttributeError):
        signals.score_candidates(frame, object())


def test_generate_signal_validates_inputs_and_handles_values():
    frame = pd.DataFrame({"edge": [2.0, 0.0, -3.0, np.inf, np.nan]})

    out = signals.generate_signal(frame, column="edge")

    assert list(out) == [1, 0, -1, 0, 0]
    with pytest.raises(ValueError, match="df cannot be empty"):
        signals.generate_signal(pd.DataFrame({"edge": []}), column="edge")
    with pytest.raises(TypeError, match="column must be a string"):
        signals.generate_signal(frame, column=cast(Any, 123))
    with pytest.raises(ValueError, match="not found"):
        signals.generate_signal(frame, column="missing")


def test_signal_matrix_cache_vote_regime_and_ensemble_helpers(monkeypatch):
    frame = _market_frame(80)
    monkeypatch.setattr(signals, "_LAST_SIGNAL_BAR", None, raising=False)
    monkeypatch.setattr(signals, "_LAST_SIGNAL_MATRIX", None, raising=False)

    matrix = signals.compute_signal_matrix(frame)
    cached = signals.compute_signal_matrix(frame)
    votes = signals.ensemble_vote_signals(
        pd.DataFrame({"a": [0.8, -0.8, 0.0], "b": [0.7, -0.7, 0.0]})
    )
    regime = signals.classify_regime(frame, window=5)

    assert not matrix.empty
    assert cached.equals(matrix)
    assert list(votes) == [1, -1, 0]
    assert set(regime.dropna().unique()).issubset({"trend", "mean_revert"})
    assert signals.generate_ensemble_signal(frame) == 1
    assert signals.compute_signal_matrix(pd.DataFrame({"close": [1.0]})).empty


def test_position_hold_and_signal_enhancement(monkeypatch):
    class FakeAction:
        HOLD = "hold_action"
        FULL_SELL = "full_sell"
        PARTIAL_SELL = "partial_sell"
        REDUCE_SIZE = "reduce"

    class FakeManager:
        def __init__(self, _ctx):
            pass

        def analyze_position(self, symbol, *_args):
            mapping = {
                "HOLD": FakeAction.HOLD,
                "SELL": FakeAction.FULL_SELL,
                "NEUTRAL": "other",
            }
            return SimpleNamespace(action=mapping[symbol])

    monkeypatch.setattr(signals, "PositionAction", FakeAction)
    monkeypatch.setattr(signals, "IntelligentPositionManager", FakeManager)

    positions = [
        SimpleNamespace(symbol="HOLD"),
        SimpleNamespace(symbol="SELL"),
        SimpleNamespace(symbol="NEUTRAL"),
        SimpleNamespace(symbol=""),
    ]
    hold_signals = signals.generate_position_hold_signals(SimpleNamespace(), positions)
    raw = [
        SimpleNamespace(symbol="HOLD", side="sell"),
        SimpleNamespace(symbol="SELL", side="buy"),
        SimpleNamespace(symbol="NEUTRAL", side="buy"),
        SimpleNamespace(symbol="", side="buy"),
    ]
    enhanced = signals.enhance_signals_with_position_logic(raw, SimpleNamespace(), hold_signals)

    assert hold_signals == {"HOLD": "hold", "SELL": "sell", "NEUTRAL": "neutral"}
    assert not signals.should_generate_new_signal("HOLD", hold_signals, {"HOLD": 1})
    assert not signals.should_generate_new_signal("AAPL", {}, {"AAPL": 2})
    assert signals.should_generate_new_signal("CASH", {}, {})
    assert [getattr(item, "symbol", "") for item in enhanced] == ["NEUTRAL", ""]


def test_market_data_preparation_correlations_positions_and_profit():
    frame = _market_frame(40)

    class Fetcher:
        def get_daily_df(self, _ctx, symbol):
            if symbol == "BAD":
                raise OSError("missing")
            return frame.copy()

    ctx = SimpleNamespace(data_fetcher=Fetcher(), portfolio_positions={"AAPL": "5", "BAD": "oops"})
    signal_list = [SimpleNamespace(symbol="AAPL", quantity=10), SimpleNamespace(symbol="BAD", quantity=3)]

    market_data = signals._prepare_market_data_for_portfolio_analysis(ctx, signal_list)
    positions = signals._get_current_portfolio_positions(ctx)
    profit = signals._estimate_signal_profit(signal_list[0], market_data)
    no_profit = signals._estimate_signal_profit(SimpleNamespace(symbol="MISSING", quantity=1), market_data)

    assert {"AAPL", "SPY"}.issubset(market_data["prices"])
    assert market_data["correlations"]["AAPL"]["AAPL"] == 1.0
    assert positions == {"AAPL": 5.0, "BAD": 0.0}
    assert profit > 0.0
    assert no_profit == 0.0


def test_signal_decision_pipeline_accepts_and_rejects_expected_cases(monkeypatch):
    frame = _market_frame(80)
    pipeline = signals.SignalDecisionPipeline(
        {"min_edge_threshold": 0.001, "ensemble_min_agree": 1, "regime_volatility_threshold": 10.0}
    )
    monkeypatch.setattr(
        signals.SignalDecisionPipeline,
        "_estimate_transaction_costs",
        lambda self, symbol, price, quantity: {"total_cost_pct": 0.0001, "total_cost": 1.0},
    )

    accepted = pipeline.evaluate_signal_with_costs("AAPL", frame, predicted_edge=0.01)
    cost_reject = pipeline.evaluate_signal_with_costs("AAPL", frame, predicted_edge=0.0001)
    low_edge = pipeline.evaluate_signal_with_costs("AAPL", frame, predicted_edge=0.0008)
    disagree = signals.SignalDecisionPipeline({"ensemble_min_agree": 3}).evaluate_signal_with_costs(
        "AAPL",
        frame,
        predicted_edge=0.01,
    )
    bad_data = pipeline.evaluate_signal_with_costs("AAPL", pd.DataFrame({"close": []}), predicted_edge=0.01)

    assert accepted["decision"] == "ACCEPT"
    assert accepted["stop_loss"] < accepted["current_price"] < accepted["take_profit"]
    assert cost_reject["reason"] == "REJECT_COST_UNPROFITABLE"
    assert low_edge["reason"] == "REJECT_EDGE_TOO_LOW"
    assert disagree["reason"] == "REJECT_ENSEMBLE_DISAGREEMENT"
    assert bad_data["reason"] == "REJECT_DATA_ERROR"


def test_generate_cost_aware_signals_uses_model_and_skips_bad_symbols(monkeypatch):
    frame = _market_frame(80)
    monkeypatch.setattr(
        signals.SignalDecisionPipeline,
        "_estimate_transaction_costs",
        lambda self, symbol, price, quantity: {"total_cost_pct": 0.0001, "total_cost": 1.0},
    )

    class Fetcher:
        def get_data(self, symbol):
            if symbol == "SHORT":
                return frame.head(10)
            if symbol == "ERR":
                raise ValueError("bad data")
            return frame.copy()

    ctx = SimpleNamespace(
        signal_pipeline_config={"ensemble_min_agree": 1, "regime_volatility_threshold": 10.0},
        data_fetcher=Fetcher(),
        feature_generator=SimpleNamespace(generate_features=lambda df: df[["close"]].tail(1)),
        model=SimpleNamespace(predict_edge=lambda _features: 0.01),
    )

    decisions = signals.generate_cost_aware_signals(ctx, ["AAPL", "SHORT", "ERR", "MSFT"])

    assert [decision["symbol"] for decision in decisions] == ["AAPL", "MSFT"]
    assert all(decision["decision"] == "ACCEPT" for decision in decisions)
