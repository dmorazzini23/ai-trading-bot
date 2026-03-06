from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from ai_trading.strategies.base import StrategySignal
from ai_trading.strategies import signals as signals_module
from ai_trading.strategies.signals import SignalAggregator, SignalProcessor


def _sig(
    symbol: str,
    side: str,
    *,
    strength: float,
    confidence: float,
    source: str = "s",
    ts: str | None = None,
) -> StrategySignal:
    metadata = {"source": source}
    if ts is not None:
        metadata["timestamp"] = ts
    return StrategySignal(
        symbol=symbol,
        side=side,
        strength=strength,
        confidence=confidence,
        signal_type="unit_test",
        metadata=metadata,
    )


def test_weighted_average_aggregation_returns_highest_weighted_group() -> None:
    agg = SignalAggregator()
    out = agg._weighted_average_aggregation(
        [
            _sig("AAPL", "buy", strength=0.8, confidence=0.5),
            _sig("AAPL", "buy", strength=0.6, confidence=1.0),
            _sig("MSFT", "sell", strength=1.0, confidence=1.0),
        ]
    )
    assert out is not None
    assert out.symbol == "MSFT"
    assert out.side == "sell"
    assert out.signal_type == "aggregated_weighted"
    assert out.metadata["source_signals"] == 1


def test_consensus_aggregation_requires_majority() -> None:
    agg = SignalAggregator()
    out = agg._consensus_aggregation(
        [
            _sig("AAPL", "buy", strength=0.6, confidence=0.9),
            _sig("AAPL", "buy", strength=0.4, confidence=0.8),
            _sig("AAPL", "sell", strength=0.9, confidence=0.9),
        ]
    )
    assert out is not None
    assert out.symbol == "AAPL"
    assert out.side == "buy"
    assert out.signal_type == "consensus"
    assert out.metadata["consensus_count"] == 2
    assert out.metadata["total_signals"] == 3


def test_resolve_signal_conflicts_majority_strongest_and_veto() -> None:
    majority = SignalAggregator(conflict_resolution="majority")
    tie = majority._resolve_signal_conflicts(
        [
            _sig("AAPL", "buy", strength=0.4, confidence=0.8),
            _sig("AAPL", "sell", strength=0.9, confidence=0.9),
        ]
    )
    assert len(tie) == 1
    assert tie[0].side == "sell"

    strongest = SignalAggregator(conflict_resolution="strongest")
    strongest_out = strongest._resolve_signal_conflicts(
        [
            _sig("AAPL", "buy", strength=0.4, confidence=0.8),
            _sig("AAPL", "sell", strength=0.9, confidence=0.9),
        ]
    )
    assert len(strongest_out) == 1
    assert strongest_out[0].side == "sell"

    veto = SignalAggregator(conflict_resolution="veto")
    veto_out = veto._resolve_signal_conflicts(
        [
            _sig("AAPL", "buy", strength=0.4, confidence=0.8),
            _sig("AAPL", "sell", strength=0.9, confidence=0.9),
        ]
    )
    assert veto_out == []


def test_apply_signal_decay_and_turnover_penalty() -> None:
    agg = SignalAggregator(turnover_penalty=0.1)
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    one_day_ago = (now - timedelta(hours=24)).isoformat()
    signal = _sig("AAPL", "buy", strength=0.8, confidence=0.8, ts=one_day_ago)

    decayed = agg._apply_signal_decay([signal], timestamp=now)
    assert len(decayed) == 1
    assert decayed[0].strength < signal.strength
    assert decayed[0].confidence < signal.confidence
    assert decayed[0].metadata["decay_factor"] == pytest.approx(0.36787944, rel=1e-3)

    signal_id = "AAPL_buy"
    agg.ensemble_history = [{"signal_id": signal_id} for _ in range(5)]
    penalized = agg._apply_turnover_penalty(signal, timestamp=now)
    assert penalized.strength == pytest.approx(signal.strength * 0.8)
    assert penalized.metadata["recent_signals_count"] == 5


def test_prepare_meta_features_and_training_data() -> None:
    agg = SignalAggregator()
    agg.signal_metrics = {
        "s1": {"recent_performance": 0.9},
        "s2": {"recent_performance": 0.7},
    }
    features = agg._prepare_meta_features(
        [_sig("AAPL", "buy", strength=0.6, confidence=0.8), _sig("AAPL", "sell", strength=0.4, confidence=0.7)],
        market_data={"volatility": 0.2, "volume_ratio": 1.5, "spread_bps": 8.0, "momentum": 0.3},
    )
    assert features is not None
    assert len(features) == 12
    assert features[6] == pytest.approx(0.0)  # agreement with 1 buy / 1 sell
    assert features[-1] == pytest.approx(0.8)

    agg.signal_performance_history = [
        {"features": [1.0, 2.0], "performance": 0.1},
        {"features": None, "performance": 0.2},
        {"features": [3.0, 4.0], "performance": 0.3},
    ]
    x, y = agg._prepare_training_data()
    assert x == [[1.0, 2.0], [3.0, 4.0]]
    assert y == [0.1, 0.3]


def test_train_meta_model_and_stacking_aggregation_with_monkeypatched_ridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DummyRidge:
        def __init__(self, alpha: float, random_state: int) -> None:
            self.alpha = alpha
            self.random_state = random_state
            self.fitted = False

        def fit(self, x: list[list[float]], y: list[float]) -> None:
            self.fitted = True
            self.x = x
            self.y = y

        def predict(self, x: list[list[float]]) -> list[float]:
            return [0.25]

    monkeypatch.setattr(signals_module, "sklearn_available", True)
    monkeypatch.setattr(
        signals_module,
        "load_sklearn_linear_model",
        lambda: SimpleNamespace(Ridge=_DummyRidge),
    )

    agg = SignalAggregator(min_performance_window=2)
    agg.signal_performance_history = [
        {"features": [1.0, 0.1], "performance": 0.2},
        {"features": [0.8, 0.3], "performance": 0.1},
        {"features": [1.2, 0.2], "performance": 0.3},
        {"features": [0.9, 0.4], "performance": 0.15},
        {"features": [1.1, 0.5], "performance": 0.25},
    ]
    agg.signal_metrics = {"s1": {"recent_performance": 0.7}, "s2": {"recent_performance": 0.6}}

    agg._train_meta_model()
    assert agg.meta_model is not None

    ts = datetime(2026, 1, 1, tzinfo=UTC)
    out = agg._stacking_aggregation(
        [
            _sig("AAPL", "buy", strength=0.6, confidence=0.7, source="s1"),
            _sig("AAPL", "buy", strength=0.4, confidence=0.8, source="s2"),
        ],
        market_data={"volatility": 0.1, "volume_ratio": 1.1, "spread_bps": 9.0, "momentum": 0.2},
        timestamp=ts,
    )
    assert out is not None
    assert out.signal_type == "stacked_meta"
    assert out.metadata["aggregation_method"] == "stacking"
    assert out.metadata["meta_prediction"] == pytest.approx(0.25)
    assert out.metadata["timestamp"] == ts.isoformat()


def test_aggregate_signals_updates_tracking_and_statistics() -> None:
    agg = SignalAggregator(enable_stacking=False)
    ts = datetime(2026, 1, 2, tzinfo=UTC)
    out = agg.aggregate_signals(
        [
            _sig("AAPL", "buy", strength=0.6, confidence=0.8),
            _sig("AAPL", "buy", strength=0.4, confidence=0.9),
        ],
        method="unknown_method",
        timestamp=ts,
    )
    assert out is not None
    assert out.signal_type == "aggregated_weighted"
    assert len(agg.ensemble_history) == 1

    for ret in [0.02, -0.01, 0.03, 0.01, 0.0, 0.02]:
        agg.update_signal_performance("AAPL_buy", actual_return=ret)

    stats = agg.get_signal_statistics()
    assert stats["total_signals_processed"] == 1
    assert stats["tracked_signal_sources"] == 1
    assert "avg_recent_performance" in stats


def test_signal_processor_filters_and_orders_by_weighted_strength() -> None:
    proc = SignalProcessor()
    processed = proc.process_signals(
        [
            _sig("AAPL", "buy", strength=0.05, confidence=0.9),  # filtered by strength
            _sig("AAPL", "buy", strength=0.6, confidence=0.2),   # filtered by confidence
            _sig("AAPL", "buy", strength=0.7, confidence=0.9),
            _sig("MSFT", "sell", strength=0.4, confidence=0.8),
        ]
    )
    assert [s.symbol for s in processed] == ["AAPL", "MSFT"]
    assert processed[0].weighted_strength >= processed[1].weighted_strength

