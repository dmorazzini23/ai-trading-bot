from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ai_trading.core import bot_engine
from ai_trading.models.contracts import (
    DAY_SLEEVE_ML_BAR_TIMEFRAME,
    DAY_SLEEVE_ML_FEATURE_COLUMNS,
    DAY_SLEEVE_ML_FEATURE_CONTRACT_VERSION,
)
from ai_trading.registry.manifest import (
    MARKET_REGIME_CLASSIFIER_ID,
    derive_market_regime_policy,
)
from ai_trading.tools import offline_replay


class _ProbabilityModel:
    classes_ = np.asarray([0, 1])

    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, _features: Any) -> np.ndarray:
        return np.asarray([[1.0 - self.probability, self.probability]])


class _BrokenModel:
    def predict_proba(self, _features: Any) -> np.ndarray:
        raise ValueError("prediction failed")


def _bundle(
    model: Any,
    *,
    market_regime_policy: dict[str, Any] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        lineage={
            "model_id": "day-model-1",
            "model_version": "v1",
            "dataset_hash": "dataset-1",
            "feature_version": DAY_SLEEVE_ML_FEATURE_CONTRACT_VERSION,
            "model_artifact_hash": "artifact-1",
        },
        selected_threshold=0.5,
        thresholds_by_regime={
            "sideways": 0.5,
            "uptrend": 0.5,
            "downtrend": 0.5,
            "volatile": 0.5,
        },
        market_regime_policy=market_regime_policy,
        governance_status="shadow",
        serving_authority="paper_only",
    )


def _market_regime_policy(now: datetime) -> dict[str, Any]:
    def row(
        *,
        support: int,
        edge: float,
        separation: float,
        profitable_ratio: float,
        qualified: bool,
    ) -> dict[str, Any]:
        return {
            "support": support,
            "mean_post_cost_net_edge_bps": edge,
            "profitable_fold_ratio": profitable_ratio,
            "ranking_separation": {
                "high_minus_low_net_edge_bps": separation,
            },
            "evidence_qualified": qualified,
        }

    return derive_market_regime_policy(
        {
            "market_regime_classifier": MARKET_REGIME_CLASSIFIER_ID,
            "aggregate": {"evidence_qualified": True},
            "config": {
                "min_trades": 75,
                "min_mean_net_edge_bps": 0.0,
                "min_ranking_separation_bps": 0.0,
                "min_profitable_fold_ratio": 0.60,
            },
            "by_market_regime": {
                "sideways": row(
                    support=40,
                    edge=6.21,
                    separation=2.5,
                    profitable_ratio=0.80,
                    qualified=True,
                ),
                "downtrend": row(
                    support=30,
                    edge=-4.0,
                    separation=-1.0,
                    profitable_ratio=0.40,
                    qualified=False,
                ),
                "volatile": row(
                    support=30,
                    edge=-8.0,
                    separation=-2.0,
                    profitable_ratio=0.20,
                    qualified=False,
                ),
            },
        },
        generated_at=now,
    )


def _frame() -> pd.DataFrame:
    index = pd.date_range("2026-07-10 14:00", periods=3, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000.0, 1100.0, 1200.0],
        },
        index=index,
    )


def _feature_row(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        [np.linspace(0.1, 1.2, len(DAY_SLEEVE_ML_FEATURE_COLUMNS))],
        columns=DAY_SLEEVE_ML_FEATURE_COLUMNS,
        index=index[-1:],
    )


def test_finalized_day_bars_exclude_forming_five_minute_bar() -> None:
    frame = _frame()

    finalized = bot_engine._finalized_day_sleeve_bars(
        frame,
        now=datetime(2026, 7, 10, 14, 12, tzinfo=UTC),
        grace_seconds=2.0,
    )

    assert list(finalized.index) == list(frame.index[:2])
    assert frame.index[-1] not in finalized.index


def test_day_sleeve_fetch_window_covers_sma_200_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 7, 10, 14, 12, tzinfo=UTC)
    monkeypatch.setenv("AI_TRADING_DAY_SLEEVE_ML_LOOKBACK_DAYS", "10")

    day_start = bot_engine._netting_sleeve_fetch_start(
        sleeve_name="day",
        timeframe="5Min",
        now=now,
    )
    other_start = bot_engine._netting_sleeve_fetch_start(
        sleeve_name="swing",
        timeframe="15Min",
        now=now,
    )

    assert (now - day_start).days == 10
    assert (now - day_start).total_seconds() >= 7 * 24 * 60 * 60
    assert (now - other_start).total_seconds() == 1800 * 50


def test_day_sleeve_fetch_window_cannot_be_configured_below_seven_days(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 7, 10, 14, 12, tzinfo=UTC)
    monkeypatch.setenv("AI_TRADING_DAY_SLEEVE_ML_LOOKBACK_DAYS", "1")

    start = bot_engine._netting_sleeve_fetch_start(
        sleeve_name="day",
        timeframe="5Min",
        now=now,
    )

    assert (now - start).days == 7


def test_day_sleeve_ml_blends_score_and_emits_advisory_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame().iloc[:2]
    shadow_rows: list[dict[str, Any]] = []
    monkeypatch.setattr(
        bot_engine,
        "build_day_sleeve_features",
        lambda _frame: _feature_row(frame.index),
    )
    monkeypatch.setattr(
        bot_engine,
        "_record_shadow_prediction",
        lambda payload: shadow_rows.append(dict(payload)),
    )
    monkeypatch.setattr(bot_engine, "_ml_shadow_quote_snapshot", lambda symbol=None: {})
    monkeypatch.setattr(bot_engine, "_ml_shadow_provider_snapshot", lambda: {})

    score, confidence, debug, error = bot_engine._score_day_sleeve_with_ml(
        symbol="AAPL",
        frame=frame,
        bundle=_bundle(_ProbabilityModel(0.8)),
        heuristic_score=0.2,
        heuristic_confidence=0.4,
        blend_weight=0.5,
        now=datetime(2026, 7, 10, 14, 12, tzinfo=UTC),
    )

    assert error is None
    assert score == pytest.approx(0.4)
    assert confidence == pytest.approx(0.5)
    assert debug is not None
    assert debug["ml_influenced"] is True
    assert debug["ml_positive_probability"] == pytest.approx(0.8)
    assert debug["ml_raw_score"] == pytest.approx(0.6)
    assert debug["ml_serving_score"] == pytest.approx(0.4)
    assert debug["ml_blend_weight"] == pytest.approx(0.5)
    assert debug["ml_serving_timeframe"] == DAY_SLEEVE_ML_BAR_TIMEFRAME
    assert debug["model_lineage"]["model_id"] == "day-model-1"
    assert debug["ml_governance_status"] == "shadow"
    assert debug["ml_serving_authority"] == "paper_only"
    assert len(shadow_rows) == 1
    shadow = shadow_rows[0]
    assert shadow["advisory"] is True
    assert shadow["executed"] is False
    assert shadow["champion_executed"] is False
    assert shadow["prediction_id"] == debug["prediction_id"]
    assert shadow["decision_id"] == debug["decision_id"]
    assert shadow["model_artifact_hash"] == "artifact-1"
    assert shadow["required_bar_timeframe"] == DAY_SLEEVE_ML_BAR_TIMEFRAME
    assert shadow["horizon_bars"] == 1
    assert shadow["market"]["entry_close"] == pytest.approx(101.5)


def test_day_sleeve_low_long_probability_abstains_instead_of_shorting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame().iloc[:2]
    monkeypatch.setattr(
        bot_engine,
        "build_day_sleeve_features",
        lambda _frame: _feature_row(frame.index),
    )
    shadow_rows: list[dict[str, Any]] = []
    monkeypatch.setattr(
        bot_engine,
        "_record_shadow_prediction",
        lambda payload: shadow_rows.append(dict(payload)),
    )
    monkeypatch.setattr(bot_engine, "_ml_shadow_quote_snapshot", lambda symbol=None: {})
    monkeypatch.setattr(bot_engine, "_ml_shadow_provider_snapshot", lambda: {})

    score, confidence, debug, error = bot_engine._score_day_sleeve_with_ml(
        symbol="AAPL",
        frame=frame,
        bundle=_bundle(_ProbabilityModel(0.2)),
        heuristic_score=0.0,
        heuristic_confidence=0.0,
        blend_weight=1.0,
        now=datetime(2026, 7, 10, 14, 12, tzinfo=UTC),
    )

    assert error is None
    assert score == 0.0
    assert confidence == 0.0
    assert debug is not None
    assert debug["ml_abstained"] is True
    assert debug["ml_raw_score"] == 0.0
    assert shadow_rows[0]["champion_side"] == "hold"
    assert shadow_rows[0]["champion_would_trade"] is False


@pytest.mark.parametrize(
    ("market_regime", "expected_allowed"),
    [
        ("sideways", True),
        ("downtrend", False),
        ("volatile", False),
    ],
)
def test_day_sleeve_market_regime_policy_has_live_replay_parity(
    monkeypatch: pytest.MonkeyPatch,
    market_regime: str,
    expected_allowed: bool,
) -> None:
    now = datetime(2026, 7, 18, 14, 12, tzinfo=UTC)
    frame = _frame().iloc[:2]
    policy = _market_regime_policy(now)
    monkeypatch.setattr(
        bot_engine,
        "build_day_sleeve_features",
        lambda _frame: _feature_row(frame.index),
    )
    monkeypatch.setattr(
        bot_engine,
        "infer_day_sleeve_regimes",
        lambda _close: (market_regime,),
    )
    monkeypatch.setattr(bot_engine, "_record_shadow_prediction", lambda _payload: None)
    monkeypatch.setattr(bot_engine, "_ml_shadow_quote_snapshot", lambda symbol=None: {})
    monkeypatch.setattr(bot_engine, "_ml_shadow_provider_snapshot", lambda: {})

    score, _confidence, debug, error = bot_engine._score_day_sleeve_with_ml(
        symbol="AAPL",
        frame=frame,
        bundle=_bundle(
            _ProbabilityModel(0.8),
            market_regime_policy=policy,
        ),
        heuristic_score=0.0,
        heuristic_confidence=0.0,
        blend_weight=1.0,
        now=now,
    )

    assert bot_engine.evaluate_market_regime_policy is (
        offline_replay.evaluate_market_regime_policy
    )
    assert error is None
    assert debug is not None
    assert debug["ml_regime_policy_declared"] is True
    assert debug["ml_regime_policy_allowed"] is expected_allowed
    assert debug["ml_regime_policy_reason"] == (
        "regime_observe" if expected_allowed else "regime_abstain"
    )
    assert debug["ml_abstained"] is (not expected_allowed)
    assert score == pytest.approx(0.6 if expected_allowed else 0.0)


def test_day_sleeve_prediction_failure_preserves_heuristic_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame().iloc[:2]
    monkeypatch.setattr(
        bot_engine,
        "build_day_sleeve_features",
        lambda _frame: _feature_row(frame.index),
    )
    shadow_rows: list[dict[str, Any]] = []
    monkeypatch.setattr(
        bot_engine,
        "_record_shadow_prediction",
        lambda payload: shadow_rows.append(dict(payload)),
    )

    score, confidence, debug, error = bot_engine._score_day_sleeve_with_ml(
        symbol="AAPL",
        frame=frame,
        bundle=_bundle(_BrokenModel()),
        heuristic_score=-0.3,
        heuristic_confidence=0.7,
        blend_weight=0.5,
        now=datetime(2026, 7, 10, 14, 12, tzinfo=UTC),
    )

    assert score == pytest.approx(-0.3)
    assert confidence == pytest.approx(0.7)
    assert debug is None
    assert error == "prediction failed"
    assert shadow_rows == []


@pytest.mark.parametrize(
    ("sleeve_name", "timeframe", "expected"),
    [
        ("day", "5Min", True),
        ("day", "5m", True),
        ("day", "1Min", False),
        ("swing", "5Min", False),
    ],
)
def test_day_sleeve_model_path_is_narrowly_scoped(
    sleeve_name: str,
    timeframe: str,
    expected: bool,
) -> None:
    assert bot_engine._is_day_sleeve_ml_timeframe(sleeve_name, timeframe) is expected


@pytest.mark.parametrize(
    ("execution_mode", "launch_profile", "paper", "base_url", "expected"),
    [
        ("paper", "paper_trade", True, "https://paper-api.alpaca.markets", True),
        ("live", "live_canary", False, "https://api.alpaca.markets", False),
        ("paper", "live_canary", True, "https://paper-api.alpaca.markets", False),
        ("sim", "paper_trade", True, "https://paper-api.alpaca.markets", False),
    ],
)
def test_shadow_model_authority_is_restricted_to_explicit_paper_profile(
    monkeypatch: pytest.MonkeyPatch,
    execution_mode: str,
    launch_profile: str,
    paper: bool,
    base_url: str,
    expected: bool,
) -> None:
    monkeypatch.setenv("AI_TRADING_PAPER_ALLOW_SHADOW_MODEL", "1")
    cfg = SimpleNamespace(
        execution_mode=execution_mode,
        launch_profile=launch_profile,
        paper=paper,
        alpaca_base_url=base_url,
    )

    assert bot_engine._paper_shadow_model_allowed(cfg) is expected


def test_shadow_model_authority_requires_explicit_enable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_TRADING_PAPER_ALLOW_SHADOW_MODEL", "0")
    cfg = SimpleNamespace(
        execution_mode="paper",
        launch_profile="paper_trade",
        paper=True,
        alpaca_base_url="https://paper-api.alpaca.markets",
    )

    assert bot_engine._paper_shadow_model_allowed(cfg) is False
