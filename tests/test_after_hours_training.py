from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
import zlib

import numpy as np
import pandas as pd
import pytest

from ai_trading.training import after_hours


pytest.importorskip("sklearn")


@pytest.fixture(autouse=True)
def _disable_no_new_data_skip_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # Most tests in this module assert training/promotion behavior and should not
    # short-circuit on persisted dataset fingerprint state.
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_SKIP_IF_NO_NEW_SIGNAL_DATA", "0")


def _write_tca(path: Path, n: int = 300) -> None:
    rows: list[dict[str, object]] = []
    for idx in range(n):
        rows.append(
            {
                "ts": f"2026-01-01T00:{idx % 60:02d}:00+00:00",
                "symbol": "AAPL" if idx % 2 == 0 else "MSFT",
                "status": "filled",
                "is_bps": float((-1) ** idx * (6 + (idx % 9))),
                "fill_latency_ms": float(50 + idx % 120),
                "partial_fill": bool(idx % 11 == 0),
                "regime_profile": "aggressive" if idx % 3 == 0 else "conservative",
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _synthetic_daily(symbol: str):
    idx = pd.date_range("2024-01-01", periods=480, freq="D", tz="UTC")
    seed = zlib.adler32(symbol.encode("utf-8")) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    drift = 0.08 if symbol == "AAPL" else 0.03
    steps = rng.normal(loc=drift, scale=1.4, size=len(idx))
    close = 120.0 + np.cumsum(steps)
    close = np.maximum(close, 5.0)
    open_ = close + rng.normal(0.0, 0.3, size=len(idx))
    high = np.maximum(open_, close) + rng.uniform(0.1, 1.2, size=len(idx))
    low = np.minimum(open_, close) - rng.uniform(0.1, 1.2, size=len(idx))
    volume = rng.integers(800_000, 2_400_000, size=len(idx)).astype(float)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


def _write_after_hours_report(
    path: Path,
    *,
    ts: datetime,
    model_name: str,
    model_id: str | None = None,
    governance_status: str | None = None,
    mean_expectancy_bps: float,
    max_drawdown_bps: float = 1000.0,
    hit_rate_stability: float,
    brier_score: float,
    regime_calibration: dict[str, object] | None = None,
    candidates: list[dict[str, object]],
) -> None:
    payload = {
        "ts": ts.isoformat(),
        "model": {
            "name": model_name,
            "model_id": model_id,
            "governance_status": governance_status,
        },
        "metrics": {
            "mean_expectancy_bps": float(mean_expectancy_bps),
            "max_drawdown_bps": float(max_drawdown_bps),
            "hit_rate_stability": float(hit_rate_stability),
            "brier_score": float(brier_score),
        },
        "candidate_metrics": candidates,
        "regime_calibration": regime_calibration or {},
        "promotion": {"status": governance_status},
    }
    path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")


def test_build_symbol_dataset_adds_derived_feature_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        after_hours,
        "_fetch_daily_bars",
        lambda symbol, _start, _end: _synthetic_daily(symbol),
    )
    dataset = after_hours._build_symbol_dataset(
        "AAPL",
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 1, tzinfo=UTC),
        cost_floor_bps=8.0,
    )
    assert not dataset.empty
    for column in (
        "signal",
        "atr_pct",
        "vwap_distance",
        "sma_spread",
        "macd_signal_gap",
        "rsi_centered",
    ):
        assert column in dataset.columns
        assert np.isfinite(dataset[column].to_numpy()).all()


def test_candidate_selection_score_penalizes_poor_calibration() -> None:
    strong = after_hours.CandidateMetrics(
        name="model_a",
        fold_count=5,
        profitable_fold_count=4,
        profitable_fold_ratio=0.8,
        support=150,
        mean_expectancy_bps=1.8,
        max_drawdown_bps=220.0,
        turnover_ratio=0.3,
        mean_hit_rate=0.56,
        hit_rate_stability=0.7,
        regime_metrics={},
        oof_probabilities=np.asarray([0.4, 0.6], dtype=float),
        brier_score=0.12,
    )
    weak = after_hours.CandidateMetrics(
        name="model_b",
        fold_count=5,
        profitable_fold_count=4,
        profitable_fold_ratio=0.8,
        support=150,
        mean_expectancy_bps=1.8,
        max_drawdown_bps=220.0,
        turnover_ratio=0.3,
        mean_hit_rate=0.56,
        hit_rate_stability=0.7,
        regime_metrics={},
        oof_probabilities=np.asarray([0.4, 0.6], dtype=float),
        brier_score=0.34,
    )

    assert after_hours._candidate_selection_score(strong) > after_hours._candidate_selection_score(
        weak
    )


def test_candidate_selection_score_honors_runtime_weight_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aggressive_expectancy = after_hours.CandidateMetrics(
        name="aggressive",
        fold_count=5,
        profitable_fold_count=3,
        profitable_fold_ratio=0.6,
        support=120,
        mean_expectancy_bps=4.5,
        max_drawdown_bps=180.0,
        turnover_ratio=0.2,
        mean_hit_rate=0.52,
        hit_rate_stability=0.5,
        regime_metrics={},
        oof_probabilities=np.asarray([0.4, 0.6], dtype=float),
        brier_score=0.45,
    )
    calibrated = after_hours.CandidateMetrics(
        name="calibrated",
        fold_count=5,
        profitable_fold_count=3,
        profitable_fold_ratio=0.6,
        support=120,
        mean_expectancy_bps=2.2,
        max_drawdown_bps=180.0,
        turnover_ratio=0.2,
        mean_hit_rate=0.52,
        hit_rate_stability=0.5,
        regime_metrics={},
        oof_probabilities=np.asarray([0.4, 0.6], dtype=float),
        brier_score=0.05,
    )
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MODEL_SELECTION_OVERRIDES_ENABLED", "0")
    baseline_aggressive = after_hours._candidate_selection_score(aggressive_expectancy)
    baseline_calibrated = after_hours._candidate_selection_score(calibrated)
    assert baseline_aggressive > baseline_calibrated
    override_path = tmp_path / "selection_overrides.json"
    override_path.write_text(
        json.dumps(
            {
                "weights": {
                    "drawdown_penalty": 0.003,
                    "turnover_penalty": 0.75,
                    "brier_penalty": 30.0,
                    "stability_weight": 0.5,
                    "support_log_weight": 0.05,
                    "profitable_fold_weight": 0.5,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MODEL_SELECTION_OVERRIDES_ENABLED", "1")
    monkeypatch.setenv(
        "AI_TRADING_AFTER_HOURS_MODEL_SELECTION_OVERRIDES_PATH",
        str(override_path),
    )
    overridden_aggressive = after_hours._candidate_selection_score(aggressive_expectancy)
    overridden_calibrated = after_hours._candidate_selection_score(calibrated)
    assert overridden_aggressive < overridden_calibrated


def test_candidate_selection_score_penalizes_profitable_fold_shortfall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MODEL_SELECTION_OVERRIDES_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MODEL_SELECTION_PROFITABLE_FOLD_TARGET", "0.7")
    monkeypatch.setenv(
        "AI_TRADING_AFTER_HOURS_MODEL_SELECTION_PROFITABLE_FOLD_SHORTFALL_PENALTY",
        "30.0",
    )
    weak_folds = after_hours.CandidateMetrics(
        name="weak_folds",
        fold_count=5,
        profitable_fold_count=1,
        profitable_fold_ratio=0.2,
        support=160,
        mean_expectancy_bps=4.0,
        max_drawdown_bps=170.0,
        turnover_ratio=0.22,
        mean_hit_rate=0.51,
        hit_rate_stability=0.58,
        regime_metrics={},
        oof_probabilities=np.asarray([0.4, 0.6], dtype=float),
        brier_score=0.12,
    )
    healthy_folds = after_hours.CandidateMetrics(
        name="healthy_folds",
        fold_count=5,
        profitable_fold_count=4,
        profitable_fold_ratio=0.8,
        support=160,
        mean_expectancy_bps=2.0,
        max_drawdown_bps=170.0,
        turnover_ratio=0.22,
        mean_hit_rate=0.51,
        hit_rate_stability=0.58,
        regime_metrics={},
        oof_probabilities=np.asarray([0.4, 0.6], dtype=float),
        brier_score=0.12,
    )
    assert after_hours._candidate_selection_score(healthy_folds) > after_hours._candidate_selection_score(
        weak_folds
    )


def test_filter_candidates_for_selection_enforces_fold_quality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MODEL_SELECTION_CONSTRAINTS_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MODEL_SELECTION_MIN_PROFITABLE_FOLDS", "3")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MODEL_SELECTION_MIN_PROFITABLE_FOLD_RATIO", "0.6")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MODEL_SELECTION_MIN_EXPECTANCY_BPS", "-50.0")
    low_quality = after_hours.CandidateMetrics(
        name="low_quality",
        fold_count=5,
        profitable_fold_count=2,
        profitable_fold_ratio=0.4,
        support=130,
        mean_expectancy_bps=4.5,
        max_drawdown_bps=210.0,
        turnover_ratio=0.30,
        mean_hit_rate=0.51,
        hit_rate_stability=0.52,
        regime_metrics={},
        oof_probabilities=np.asarray([0.4, 0.6], dtype=float),
        brier_score=0.14,
    )
    high_quality = after_hours.CandidateMetrics(
        name="high_quality",
        fold_count=5,
        profitable_fold_count=4,
        profitable_fold_ratio=0.8,
        support=130,
        mean_expectancy_bps=2.0,
        max_drawdown_bps=210.0,
        turnover_ratio=0.30,
        mean_hit_rate=0.51,
        hit_rate_stability=0.52,
        regime_metrics={},
        oof_probabilities=np.asarray([0.4, 0.6], dtype=float),
        brier_score=0.14,
    )
    filtered, diagnostics = after_hours._filter_candidates_for_selection(
        [low_quality, high_quality]
    )
    assert [item.name for item in filtered] == ["high_quality"]
    assert diagnostics["eligible_count"] == 1
    assert diagnostics["rejected_by_reason"]["profitable_folds"] == 1
    assert diagnostics["rejected_by_reason"]["profitable_fold_ratio"] == 1


def test_model_quality_diagnostics_reports_deciles_and_symbol_contributions() -> None:
    dataset = pd.DataFrame(
        {
            "label": np.asarray([1, 0, 1, 0, 1, 0, 1, 0], dtype=int),
            "realized_edge_bps": np.asarray([8.0, -6.0, 5.0, -4.0, 3.0, -2.5, 1.5, -1.0]),
            "symbol": ["AAPL", "AAPL", "MSFT", "MSFT", "NVDA", "NVDA", "AMD", "AMD"],
        }
    )
    probs = np.asarray([0.91, 0.87, 0.74, 0.62, 0.55, 0.44, 0.31, 0.22], dtype=float)

    diagnostics = after_hours._model_quality_diagnostics(
        dataset,
        oof_probabilities=probs,
        selected_threshold=0.5,
        deciles=4,
        top_n_symbols=10,
    )

    assert diagnostics["valid_oof_samples"] == 8
    assert diagnostics["selected_support"] == 5
    assert len(diagnostics["score_decile_expectancy"]) == 4
    calibration = diagnostics["calibration_by_decile"]
    assert 0.0 <= float(calibration["overall_ece"]) <= 1.0
    assert len(calibration["deciles"]) == 4
    symbol_contrib = diagnostics["negative_expectancy_contribution_by_symbol"]
    assert float(symbol_contrib["total_negative_net_bps"]) >= 0.0
    assert isinstance(symbol_contrib["contributors"], list)


def test_predict_probabilities_respects_positive_class_column_order() -> None:
    class _DummyModel:
        classes_ = np.asarray([1, 0], dtype=int)

        @staticmethod
        def predict_proba(_X) -> np.ndarray:
            return np.asarray(
                [
                    [0.91, 0.09],
                    [0.73, 0.27],
                    [0.52, 0.48],
                ],
                dtype=float,
            )

    probs = after_hours._predict_probabilities(_DummyModel(), np.asarray([[0.0], [1.0], [2.0]]))
    assert np.allclose(probs, np.asarray([0.91, 0.73, 0.52], dtype=float))


def test_segment_reweight_sample_weights_targets_high_conf_negative_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_SEGMENT_REWEIGHT_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_SEGMENT_REWEIGHT_HIGH_CONF_QUANTILE", "0.75")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_SEGMENT_REWEIGHT_HIGH_CONF_MULTIPLIER", "1.8")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_SEGMENT_REWEIGHT_MIN_SYMBOL_SUPPORT", "2")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_SEGMENT_REWEIGHT_MIN_NEGATIVE_SHARE", "0.05")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_SEGMENT_REWEIGHT_SYMBOL_MULTIPLIER", "1.6")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_SEGMENT_REWEIGHT_SYMBOL_MULTIPLIER_MAX", "2.6")

    dataset = pd.DataFrame(
        {
            "label": np.asarray([0, 0, 0, 1, 1, 0, 1, 0], dtype=int),
            "realized_edge_bps": np.asarray([-80.0, -55.0, -30.0, 12.0, 8.0, -10.0, 5.0, -6.0]),
            "symbol": ["ORCL", "ORCL", "AVGO", "AVGO", "AAPL", "AAPL", "MSFT", "MSFT"],
        }
    )
    probs = np.asarray([0.94, 0.91, 0.88, 0.86, 0.62, 0.59, 0.54, 0.51], dtype=float)

    weights, report = after_hours._build_segment_reweight_sample_weights(
        dataset,
        oof_probabilities=probs,
        selected_threshold=0.5,
    )

    assert report["applied"] is True
    assert report["high_conf_negative_count"] >= 2
    assert report["symbol_adjusted_rows"] >= 2
    assert report["symbol_adjustments"]
    assert np.max(weights) > 1.0
    # ORCL rows should be upweighted because they dominate negative contribution.
    assert weights[0] > 1.0
    assert weights[1] > 1.0


def test_candidate_threshold_selection_score_penalizes_turnover_and_downside(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_CANDIDATE_THRESHOLD_TURNOVER_PENALTY", "1.0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_CANDIDATE_THRESHOLD_WORST_FOLD_WEIGHT", "0.2")
    monkeypatch.setenv(
        "AI_TRADING_AFTER_HOURS_CANDIDATE_THRESHOLD_DOWNSIDE_SEMIDEV_PENALTY",
        "0.15",
    )

    stable_metrics = {
        "mean_expectancy_bps": 2.0,
        "max_drawdown_bps": 200.0,
        "turnover_ratio": 0.20,
        "mean_hit_rate": 0.56,
        "profitable_fold_ratio": 0.75,
        "hit_rate_stability": 0.68,
        "support": 150,
        "fold_expectancy_bps": [2.4, 2.1, 1.8, 1.6, 1.9],
    }
    unstable_metrics = {
        "mean_expectancy_bps": 2.0,
        "max_drawdown_bps": 200.0,
        "turnover_ratio": 0.55,
        "mean_hit_rate": 0.56,
        "profitable_fold_ratio": 0.75,
        "hit_rate_stability": 0.68,
        "support": 150,
        "fold_expectancy_bps": [6.0, 5.0, -8.5, 4.5, -4.0],
    }

    assert after_hours._candidate_threshold_selection_score(
        stable_metrics
    ) > after_hours._candidate_threshold_selection_score(unstable_metrics)


def test_model_selection_retune_skips_without_drift_breach(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    now = datetime(2026, 1, 10, 21, 15, tzinfo=UTC)
    for idx in range(8):
        ts = now - timedelta(days=8 - idx)
        _write_after_hours_report(
            report_dir / f"after_hours_training_202601{idx + 1:02d}.json",
            ts=ts,
            model_name="logreg",
            mean_expectancy_bps=18.0,
            hit_rate_stability=0.79,
            brier_score=0.16,
            candidates=[
                {
                    "name": "logreg",
                    "selected": True,
                    "mean_expectancy_bps": 18.0,
                    "max_drawdown_bps": 240.0,
                    "turnover_ratio": 0.18,
                    "hit_rate_stability": 0.79,
                    "brier_score": 0.16,
                    "support": 160,
                    "profitable_fold_ratio": 0.8,
                },
                {
                    "name": "xgboost",
                    "selected": False,
                    "mean_expectancy_bps": 16.0,
                    "max_drawdown_bps": 220.0,
                    "turnover_ratio": 0.16,
                    "hit_rate_stability": 0.8,
                    "brier_score": 0.15,
                    "support": 160,
                    "profitable_fold_ratio": 0.82,
                },
            ],
        )
    override_path = tmp_path / "runtime" / "model_selection_overrides.json"
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RETUNE_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RETUNE_MIN_REPORTS", "6")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RETUNE_BASELINE_WINDOW", "4")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RETUNE_RECENT_WINDOW", "2")
    monkeypatch.setenv(
        "AI_TRADING_AFTER_HOURS_MODEL_SELECTION_OVERRIDES_PATH",
        str(override_path),
    )
    summary = after_hours._maybe_retune_model_selection_weights(
        now_utc=now,
        report_dir=report_dir,
        pending_report=None,
        current_weights=after_hours._resolved_model_selection_weights(),
    )
    assert summary["retuned"] is False
    assert summary["reason"] == "no_drift_breach"
    assert not override_path.exists()


def test_model_selection_retune_writes_overrides_on_drift_breach(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    now = datetime(2026, 1, 10, 21, 15, tzinfo=UTC)
    for idx in range(9):
        ts = now - timedelta(days=9 - idx)
        degraded = idx >= 6
        _write_after_hours_report(
            report_dir / f"after_hours_training_202602{idx + 1:02d}.json",
            ts=ts,
            model_name="logreg",
            mean_expectancy_bps=(8.0 if degraded else 20.0),
            hit_rate_stability=(0.62 if degraded else 0.82),
            brier_score=(0.34 if degraded else 0.11),
            candidates=[
                {
                    "name": "logreg",
                    "selected": True,
                    "mean_expectancy_bps": 20.0,
                    "max_drawdown_bps": 450.0,
                    "turnover_ratio": 0.25,
                    "hit_rate_stability": 0.4,
                    "brier_score": 0.5,
                    "support": 120,
                    "profitable_fold_ratio": 0.4,
                },
                {
                    "name": "xgboost",
                    "selected": False,
                    "mean_expectancy_bps": 14.0,
                    "max_drawdown_bps": 150.0,
                    "turnover_ratio": 0.1,
                    "hit_rate_stability": 0.8,
                    "brier_score": 0.05,
                    "support": 120,
                    "profitable_fold_ratio": 0.8,
                },
            ],
        )
    override_path = tmp_path / "runtime" / "model_selection_overrides.json"
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RETUNE_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RETUNE_MIN_REPORTS", "8")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RETUNE_BASELINE_WINDOW", "5")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RETUNE_RECENT_WINDOW", "3")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RETUNE_BRIER_DRIFT_PCT", "0.1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RETUNE_EXPECTANCY_DROP_BPS", "4.0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RETUNE_STABILITY_DROP", "0.05")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RETUNE_EVAL_DRAWDOWN_PENALTY", "0.02")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RETUNE_EVAL_BRIER_PENALTY", "30.0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RETUNE_EVAL_TURNOVER_PENALTY", "1.0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RETUNE_WEIGHT_REGULARIZATION", "0.0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RETUNE_MIN_UTILITY_IMPROVEMENT", "0.0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RETUNE_MIN_WEIGHT_CHANGE_PCT", "0.0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RETUNE_SEARCH_FACTORS", "0.5,1.0,1.5,2.0,3.0")
    monkeypatch.setenv(
        "AI_TRADING_AFTER_HOURS_MODEL_SELECTION_OVERRIDES_PATH",
        str(override_path),
    )
    summary = after_hours._maybe_retune_model_selection_weights(
        now_utc=now,
        report_dir=report_dir,
        pending_report=None,
        current_weights=after_hours._resolved_model_selection_weights(),
    )
    assert summary["retuned"] is True
    assert summary["reason"] == "drift_breach"
    assert Path(summary["override_path"]).exists()
    payload = json.loads(Path(summary["override_path"]).read_text(encoding="utf-8"))
    assert "weights" in payload
    assert payload["weights"] != after_hours._model_selection_default_weights()


def test_model_selection_retune_respects_cooldown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    now = datetime(2026, 1, 10, 21, 15, tzinfo=UTC)
    for idx in range(9):
        ts = now - timedelta(days=9 - idx)
        degraded = idx >= 6
        _write_after_hours_report(
            report_dir / f"after_hours_training_202603{idx + 1:02d}.json",
            ts=ts,
            model_name="logreg",
            mean_expectancy_bps=(8.0 if degraded else 20.0),
            hit_rate_stability=(0.62 if degraded else 0.82),
            brier_score=(0.34 if degraded else 0.11),
            candidates=[
                {
                    "name": "logreg",
                    "selected": True,
                    "mean_expectancy_bps": 20.0,
                    "max_drawdown_bps": 450.0,
                    "turnover_ratio": 0.25,
                    "hit_rate_stability": 0.4,
                    "brier_score": 0.5,
                    "support": 120,
                    "profitable_fold_ratio": 0.4,
                },
                {
                    "name": "xgboost",
                    "selected": False,
                    "mean_expectancy_bps": 14.0,
                    "max_drawdown_bps": 150.0,
                    "turnover_ratio": 0.1,
                    "hit_rate_stability": 0.8,
                    "brier_score": 0.05,
                    "support": 120,
                    "profitable_fold_ratio": 0.8,
                },
            ],
        )
    override_path = tmp_path / "runtime" / "model_selection_overrides.json"
    override_path.parent.mkdir(parents=True, exist_ok=True)
    override_path.write_text(
        json.dumps(
            {
                "updated_at": (now - timedelta(hours=1)).isoformat(),
                "weights": after_hours._model_selection_default_weights(),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RETUNE_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RETUNE_MIN_REPORTS", "8")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RETUNE_BASELINE_WINDOW", "5")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RETUNE_RECENT_WINDOW", "3")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RETUNE_BRIER_DRIFT_PCT", "0.1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RETUNE_EXPECTANCY_DROP_BPS", "4.0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RETUNE_STABILITY_DROP", "0.05")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RETUNE_EVAL_DRAWDOWN_PENALTY", "0.02")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RETUNE_EVAL_BRIER_PENALTY", "30.0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RETUNE_WEIGHT_REGULARIZATION", "0.0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RETUNE_MIN_UTILITY_IMPROVEMENT", "0.0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RETUNE_MIN_WEIGHT_CHANGE_PCT", "0.0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RETUNE_SEARCH_FACTORS", "0.5,1.0,1.5,2.0,3.0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RETUNE_MIN_HOURS_BETWEEN_UPDATES", "24")
    monkeypatch.setenv(
        "AI_TRADING_AFTER_HOURS_MODEL_SELECTION_OVERRIDES_PATH",
        str(override_path),
    )
    summary = after_hours._maybe_retune_model_selection_weights(
        now_utc=now,
        report_dir=report_dir,
        pending_report=None,
        current_weights=after_hours._resolved_model_selection_weights(),
    )
    assert summary["retuned"] is False
    assert summary["reason"] == "cooldown_active"


def test_selection_weight_safety_envelope_clamps_aggressive_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = after_hours._model_selection_default_weights()
    candidate = {key: value * 10.0 for key, value in current.items()}
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RETUNE_MAX_ABS_WEIGHT_DELTA", "0.1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RETUNE_MAX_RELATIVE_WEIGHT_DELTA", "0.1")
    result = after_hours._apply_selection_weight_safety_envelope(
        current_weights=current,
        candidate_weights=candidate,
    )
    assert result["clamped"] is True
    for key, new_value in result["weights"].items():
        assert abs(float(new_value) - float(current[key])) <= 0.1 + 1e-9


def test_regime_calibration_threshold_adjustment_applies_bump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_REGIME_CALIBRATION_THRESHOLD_ADJUST_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_REGIME_ECE_WARN", "0.08")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_REGIME_ECE_CRITICAL", "0.12")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_REGIME_THRESHOLD_BUMP_STEP", "0.02")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_REGIME_THRESHOLD_MAX_BUMP", "0.08")
    adjusted, adjustments = after_hours._apply_regime_calibration_threshold_adjustment(
        thresholds_by_regime={"uptrend": 0.5, "sideways": 0.4},
        regime_calibration={
            "uptrend": {"ece": 0.16},
            "sideways": {"ece": 0.04},
        },
    )
    assert adjusted["uptrend"] > 0.5
    assert adjusted["sideways"] == pytest.approx(0.4)
    assert "uptrend" in adjustments


def test_load_prior_model_metrics_prefers_distinct_production_model(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    base_ts = datetime(2026, 1, 10, 21, 15, tzinfo=UTC)
    _write_after_hours_report(
        report_dir / "after_hours_training_20260110_211500.json",
        ts=base_ts - timedelta(days=2),
        model_name="logreg",
        model_id="model-prod-a",
        governance_status="production",
        mean_expectancy_bps=10.0,
        max_drawdown_bps=700.0,
        hit_rate_stability=0.7,
        brier_score=0.2,
        candidates=[
            {
                "name": "logreg",
                "selected": True,
                "mean_expectancy_bps": 10.0,
                "max_drawdown_bps": 700.0,
                "turnover_ratio": 0.2,
                "hit_rate_stability": 0.7,
                "brier_score": 0.2,
                "support": 120,
                "profitable_fold_ratio": 0.6,
            }
        ],
    )
    _write_after_hours_report(
        report_dir / "after_hours_training_20260110_221500.json",
        ts=base_ts - timedelta(days=1),
        model_name="logreg",
        model_id="model-shadow-b",
        governance_status="shadow",
        mean_expectancy_bps=12.0,
        max_drawdown_bps=900.0,
        hit_rate_stability=0.71,
        brier_score=0.19,
        candidates=[
            {
                "name": "logreg",
                "selected": True,
                "mean_expectancy_bps": 12.0,
                "max_drawdown_bps": 900.0,
                "turnover_ratio": 0.2,
                "hit_rate_stability": 0.71,
                "brier_score": 0.19,
                "support": 120,
                "profitable_fold_ratio": 0.6,
            }
        ],
    )
    # Most recent report has same model id as current latest daily alias;
    # loader should still look back to a distinct prior promoted model.
    _write_after_hours_report(
        report_dir / "after_hours_training_20260110.json",
        ts=base_ts,
        model_name="logreg",
        model_id="model-shadow-b",
        governance_status="shadow",
        mean_expectancy_bps=12.0,
        max_drawdown_bps=900.0,
        hit_rate_stability=0.71,
        brier_score=0.19,
        candidates=[
            {
                "name": "logreg",
                "selected": True,
                "mean_expectancy_bps": 12.0,
                "max_drawdown_bps": 900.0,
                "turnover_ratio": 0.2,
                "hit_rate_stability": 0.71,
                "brier_score": 0.19,
                "support": 120,
                "profitable_fold_ratio": 0.6,
            }
        ],
    )

    prior = after_hours._load_prior_model_metrics(report_dir=report_dir)
    assert prior is not None
    assert prior["model_id"] == "model-prod-a"
    assert prior["governance_status"] == "production"


def test_after_hours_training_skips_before_close() -> None:
    result = after_hours.run_after_hours_training(
        now=datetime(2026, 1, 6, 19, 0, tzinfo=UTC),  # 14:00 New York
    )
    assert result["status"] == "skipped"
    assert result["reason"] == "before_market_close"


def test_after_hours_training_allows_overnight_catchup_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_CATCHUP_ENABLED", "1")
    monkeypatch.setattr(after_hours, "_load_symbols", lambda: [])

    result = after_hours.run_after_hours_training(
        now=datetime(2026, 1, 7, 5, 10, tzinfo=UTC),  # 00:10 New York
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "no_symbols"


def test_after_hours_training_skips_overnight_when_catchup_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_CATCHUP_ENABLED", "0")
    monkeypatch.setattr(after_hours, "_load_symbols", lambda: [])

    result = after_hours.run_after_hours_training(
        now=datetime(2026, 1, 7, 5, 10, tzinfo=UTC),  # 00:10 New York
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "before_market_close"


def test_after_hours_training_trains_and_writes_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tickers = tmp_path / "tickers.csv"
    tickers.write_text("symbol\nAAPL\nMSFT\n", encoding="utf-8")
    tca_path = tmp_path / "tca_records.jsonl"
    _write_tca(tca_path, n=420)

    monkeypatch.setenv("AI_TRADING_TICKERS_FILE", str(tickers))
    monkeypatch.setenv("AI_TRADING_TCA_PATH", str(tca_path))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_REPORT_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("AI_TRADING_MODEL_PATH", str(tmp_path / "runtime_model.joblib"))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTE_MODEL_PATH", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MIN_ROWS", "120")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_CV_SPLITS", "3")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_LOOKBACK_DAYS", "420")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RL_OVERLAY_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_AUTO_PROMOTE", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MIN_THRESHOLD_SUPPORT", "8")
    monkeypatch.setattr(
        after_hours,
        "_fetch_daily_bars",
        lambda symbol, _start, _end: _synthetic_daily(symbol),
    )

    result = after_hours.run_after_hours_training(
        now=datetime(2026, 1, 6, 21, 10, tzinfo=UTC),  # 16:10 New York
    )
    assert result["status"] == "trained"
    assert Path(result["model_path"]).exists()
    assert Path(result["manifest_path"]).exists()
    assert Path(result["report_path"]).exists()
    assert Path(result["daily_report_path"]).exists()
    if result["promoted_model_path"] is not None:
        assert Path(result["promoted_model_path"]).exists()
    if result["promoted_manifest_path"] is not None:
        assert Path(result["promoted_manifest_path"]).exists()
    assert "_" in Path(result["report_path"]).stem

    report_payload = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
    assert "metrics" in report_payload
    assert "thresholds_by_regime" in report_payload
    assert "candidate_metrics" in report_payload
    assert "sensitivity_sweep" in report_payload
    assert "model_quality" in report_payload
    assert "manifest_metadata" in report_payload["model"]
    assert "hard_negative_mining" in report_payload
    assert "segment_reweighting" in report_payload
    assert "sample_weighting" in report_payload
    assert "edge_model_v2" in report_payload
    assert isinstance(report_payload["candidate_metrics"], list)
    assert report_payload["candidate_metrics"]
    assert "candidate_metrics" in result
    assert isinstance(result["candidate_metrics"], list)
    assert result["candidate_metrics"]
    assert "sensitivity_sweep" in result
    assert "model_quality" in result
    assert "hard_negative_mining" in result
    assert "segment_reweighting" in result
    assert "sample_weighting" in result
    assert "edge_model_v2" in result
    assert "score_decile_expectancy" in report_payload["model_quality"]
    assert "calibration_by_decile" in report_payload["model_quality"]
    assert "negative_expectancy_contribution_by_symbol" in report_payload["model_quality"]

    import joblib

    trained_model = joblib.load(result["model_path"])
    assert hasattr(trained_model, "hard_negative_weighted_fit_")
    assert hasattr(trained_model, "edge_model_v2_bundle_")
    assert isinstance(getattr(trained_model, "edge_model_v2_bundle_"), dict)

    manifest_payload = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert "metadata" in manifest_payload
    assert manifest_payload["metadata"]["strategy"] == "after_hours_ml_edge"
    assert "segment_reweighting" in manifest_payload["metadata"]
    assert "sample_weighting" in manifest_payload["metadata"]


def test_after_hours_training_skips_when_no_new_signal_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tickers = tmp_path / "tickers.csv"
    tickers.write_text("symbol\nAAPL\nMSFT\n", encoding="utf-8")
    tca_path = tmp_path / "tca_records.jsonl"
    _write_tca(tca_path, n=420)
    state_path = tmp_path / "runtime" / "after_hours_training_state.json"

    monkeypatch.setenv("AI_TRADING_TICKERS_FILE", str(tickers))
    monkeypatch.setenv("AI_TRADING_TCA_PATH", str(tca_path))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_REPORT_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_STATE_PATH", str(state_path))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTE_MODEL_PATH", "0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MIN_ROWS", "120")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_CV_SPLITS", "3")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_LOOKBACK_DAYS", "420")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RL_OVERLAY_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_AUTO_PROMOTE", "0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MIN_THRESHOLD_SUPPORT", "8")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_SKIP_IF_NO_NEW_SIGNAL_DATA", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MIN_NEW_ROWS_FOR_RETRAIN", "25")
    monkeypatch.setattr(
        after_hours,
        "_fetch_daily_bars",
        lambda symbol, _start, _end: _synthetic_daily(symbol),
    )

    first = after_hours.run_after_hours_training(
        now=datetime(2026, 1, 6, 21, 10, tzinfo=UTC),
    )
    assert first["status"] == "trained"
    assert state_path.exists()

    second = after_hours.run_after_hours_training(
        now=datetime(2026, 1, 6, 21, 15, tzinfo=UTC),
    )
    assert second["status"] == "skipped"
    assert second["reason"] == "no_new_signal_data"
    assert second["new_rows"] < second["min_new_rows"]
    assert second["unchanged_dataset_fingerprint"] is True


def test_after_hours_training_state_strips_pytest_temp_report_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path = tmp_path / "runtime" / "after_hours_training_state.json"
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_STATE_PATH", str(state_path))

    written = after_hours._write_after_hours_training_state(
        {
            "updated_at": "2026-01-06T21:10:00+00:00",
            "rows": 560,
            "dataset_fingerprint": "fp-001",
            "max_label_ts": "2025-04-24T00:00:00+00:00",
            "report_path": "/tmp/pytest-of-root/pytest-77/reports/after_hours_training_20260106_211000.json",
            "daily_report_path": "/tmp/pytest-of-root/pytest-77/reports/after_hours_training_20260106.json",
            "model_id": "ml-edge-001",
            "model_name": "histgb",
        }
    )

    assert written is not None
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert "report_path" not in payload
    assert "daily_report_path" not in payload
    assert payload["model_id"] == "ml-edge-001"


def test_after_hours_training_state_prefers_report_root_state_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preferred_state_path = tmp_path / "report_runtime" / "after_hours_training_state.json"
    monkeypatch.delenv("AI_TRADING_AFTER_HOURS_TRAINING_STATE_PATH", raising=False)

    written = after_hours._write_after_hours_training_state(
        {
            "updated_at": "2026-01-06T21:10:00+00:00",
            "rows": 560,
            "dataset_fingerprint": "fp-001",
            "max_label_ts": "2025-04-24T00:00:00+00:00",
            "model_id": "ml-edge-001",
            "model_name": "histgb",
        },
        preferred_path=preferred_state_path,
    )

    assert written is not None
    assert Path(written) == preferred_state_path.resolve()
    assert preferred_state_path.exists()

    loaded = after_hours._load_after_hours_training_state(
        preferred_path=preferred_state_path
    )
    assert loaded is not None
    assert loaded["model_id"] == "ml-edge-001"


def test_after_hours_training_falls_back_when_model_dir_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    tickers = tmp_path / "tickers.csv"
    tickers.write_text("symbol\nAAPL\nMSFT\n", encoding="utf-8")
    tca_path = tmp_path / "tca_records.jsonl"
    _write_tca(tca_path, n=420)

    requested_model_dir = tmp_path / "readonly_models"
    requested_model_dir.mkdir(parents=True, exist_ok=True)
    requested_model_dir.chmod(0o555)
    fallback_models_root = tmp_path / "fallback_models_root"
    monkeypatch.setattr(after_hours.paths, "MODELS_DIR", fallback_models_root)
    original_access = after_hours.os.access
    monkeypatch.setattr(
        after_hours.os,
        "access",
        lambda candidate, mode: (
            False
            if Path(candidate).resolve() == requested_model_dir.resolve()
            else original_access(candidate, mode)
        ),
    )

    monkeypatch.setenv("AI_TRADING_TICKERS_FILE", str(tickers))
    monkeypatch.setenv("AI_TRADING_TCA_PATH", str(tca_path))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MODEL_DIR", str(requested_model_dir))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_REPORT_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTE_MODEL_PATH", "0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MIN_ROWS", "120")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_CV_SPLITS", "3")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_LOOKBACK_DAYS", "420")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RL_OVERLAY_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_AUTO_PROMOTE", "0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MIN_THRESHOLD_SUPPORT", "8")
    monkeypatch.setattr(
        after_hours,
        "_fetch_daily_bars",
        lambda symbol, _start, _end: _synthetic_daily(symbol),
    )
    caplog.set_level("WARNING")

    try:
        result = after_hours.run_after_hours_training(
            now=datetime(2026, 1, 6, 21, 10, tzinfo=UTC),  # 16:10 New York
        )
    finally:
        requested_model_dir.chmod(0o755)

    assert result["status"] == "trained"
    resolved_model_path = Path(result["model_path"])
    assert resolved_model_path.exists()
    assert resolved_model_path.parent == (fallback_models_root / "after_hours")
    assert any(
        record.message == "AFTER_HOURS_MODEL_DIR_FALLBACK"
        for record in caplog.records
    )


def test_after_hours_sensitivity_gate_can_block_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tickers = tmp_path / "tickers.csv"
    tickers.write_text("symbol\nAAPL\nMSFT\n", encoding="utf-8")
    tca_path = tmp_path / "tca_records.jsonl"
    _write_tca(tca_path, n=420)

    monkeypatch.setenv("AI_TRADING_TICKERS_FILE", str(tickers))
    monkeypatch.setenv("AI_TRADING_TCA_PATH", str(tca_path))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_REPORT_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MIN_ROWS", "120")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_CV_SPLITS", "3")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_LOOKBACK_DAYS", "420")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MIN_THRESHOLD_SUPPORT", "8")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RL_OVERLAY_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_AUTO_PROMOTE", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_SENSITIVITY_SWEEP_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_SWEEP_MIN_SCENARIO_EXPECTANCY_BPS", "9999")
    monkeypatch.setattr(
        after_hours,
        "_fetch_daily_bars",
        lambda symbol, _start, _end: _synthetic_daily(symbol),
    )

    result = after_hours.run_after_hours_training(
        now=datetime(2026, 1, 6, 21, 10, tzinfo=UTC),
    )
    assert result["status"] == "trained"
    assert result["governance_status"] == "shadow"
    assert result["edge_gates"]["sensitivity"] is False
    assert result["sensitivity_sweep"]["enabled"] is True
    assert result["sensitivity_sweep"]["gate"] is False


def test_after_hours_runtime_promotion_skips_shadow_governance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tickers = tmp_path / "tickers.csv"
    tickers.write_text("symbol\nAAPL\nMSFT\n", encoding="utf-8")
    tca_path = tmp_path / "tca_records.jsonl"
    _write_tca(tca_path, n=420)

    monkeypatch.setenv("AI_TRADING_TICKERS_FILE", str(tickers))
    monkeypatch.setenv("AI_TRADING_TCA_PATH", str(tca_path))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_REPORT_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("AI_TRADING_MODEL_PATH", str(tmp_path / "runtime_model.joblib"))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTE_MODEL_PATH", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_AUTO_PROMOTE", "0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MIN_ROWS", "120")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_CV_SPLITS", "3")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_LOOKBACK_DAYS", "420")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MIN_THRESHOLD_SUPPORT", "8")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RL_OVERLAY_ENABLED", "0")
    monkeypatch.setattr(
        after_hours,
        "_fetch_daily_bars",
        lambda symbol, _start, _end: _synthetic_daily(symbol),
    )

    result = after_hours.run_after_hours_training(
        now=datetime(2026, 1, 6, 21, 10, tzinfo=UTC),
    )

    assert result["status"] == "trained"
    assert result["governance_status"] == "shadow"
    assert result["promoted_model_path"] is None
    assert result["promoted_manifest_path"] is None


def test_after_hours_strict_promotion_policy_blocks_when_min_rows_not_met(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tickers = tmp_path / "tickers.csv"
    tickers.write_text("symbol\nAAPL\nMSFT\n", encoding="utf-8")
    tca_path = tmp_path / "tca_records.jsonl"
    _write_tca(tca_path, n=420)

    monkeypatch.setenv("AI_TRADING_TICKERS_FILE", str(tickers))
    monkeypatch.setenv("AI_TRADING_TCA_PATH", str(tca_path))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_REPORT_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MIN_ROWS", "120")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_CV_SPLITS", "3")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_LOOKBACK_DAYS", "420")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MIN_THRESHOLD_SUPPORT", "8")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RL_OVERLAY_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_AUTO_PROMOTE", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_POLICY", "strict")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_ROWS", "999999")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_SUPPORT", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_FOLDS", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_HIT_RATE", "0.0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_SENSITIVITY_SWEEP_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_EDGE_TARGET_EXPECTANCY_BPS", "-9999")
    monkeypatch.setenv("AI_TRADING_EDGE_TARGET_MAX_DRAWDOWN_BPS", "999999")
    monkeypatch.setenv("AI_TRADING_EDGE_TARGET_MAX_TURNOVER_RATIO", "1.0")
    monkeypatch.setenv("AI_TRADING_EDGE_TARGET_MIN_HIT_RATE_STABILITY", "0.0")
    monkeypatch.setattr(
        after_hours,
        "_fetch_daily_bars",
        lambda symbol, _start, _end: _synthetic_daily(symbol),
    )

    result = after_hours.run_after_hours_training(
        now=datetime(2026, 1, 6, 21, 10, tzinfo=UTC),
    )

    assert result["status"] == "trained"
    assert result["governance_status"] == "shadow"
    assert result["promotion"]["policy"] == "strict"
    assert result["promotion"]["strict_gates"]["rows"] is False


def test_after_hours_strict_promotion_policy_can_promote_when_all_gates_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tickers = tmp_path / "tickers.csv"
    tickers.write_text("symbol\nAAPL\nMSFT\n", encoding="utf-8")
    tca_path = tmp_path / "tca_records.jsonl"
    _write_tca(tca_path, n=420)

    monkeypatch.setenv("AI_TRADING_TICKERS_FILE", str(tickers))
    monkeypatch.setenv("AI_TRADING_TCA_PATH", str(tca_path))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_REPORT_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MIN_ROWS", "120")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_CV_SPLITS", "3")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_LOOKBACK_DAYS", "420")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MIN_THRESHOLD_SUPPORT", "8")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RL_OVERLAY_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_AUTO_PROMOTE", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_POLICY", "strict")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_ROWS", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_SUPPORT", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_FOLDS", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_HIT_RATE", "0.0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_PROFITABLE_FOLDS", "0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_PROFITABLE_FOLD_RATIO", "0.0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_REQUIRE_PRIOR_IMPROVEMENT", "0")
    monkeypatch.setenv("AI_TRADING_POLICY_PROMOTION_MIN_OOS_SAMPLES", "1")
    monkeypatch.setenv("AI_TRADING_POLICY_PROMOTION_MIN_OOS_NET_BPS", "-9999")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_SENSITIVITY_SWEEP_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_EDGE_TARGET_EXPECTANCY_BPS", "-9999")
    monkeypatch.setenv("AI_TRADING_EDGE_TARGET_MAX_DRAWDOWN_BPS", "999999")
    monkeypatch.setenv("AI_TRADING_EDGE_TARGET_MAX_TURNOVER_RATIO", "1.0")
    monkeypatch.setenv("AI_TRADING_EDGE_TARGET_MIN_HIT_RATE_STABILITY", "0.0")
    monkeypatch.setattr(
        after_hours,
        "_fetch_daily_bars",
        lambda symbol, _start, _end: _synthetic_daily(symbol),
    )

    result = after_hours.run_after_hours_training(
        now=datetime(2026, 1, 6, 21, 10, tzinfo=UTC),
    )

    assert result["status"] == "trained"
    assert result["governance_status"] == "production"
    assert result["promotion"]["policy"] == "strict"
    assert result["promotion"]["gate_passed"] is True
    assert "roadmap" in result
    assert "phase_1_week_1" in result["roadmap"]


def test_promotion_consecutive_pass_gate_requires_second_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_CONSECUTIVE_PASSES", "2")
    base_promotion = {
        "auto_promote": True,
        "gate_passed": True,
        "status": "production",
        "combined_gates": {"expectancy": True},
        "additional_gates": {},
    }
    first_day = datetime(2026, 1, 6, tzinfo=UTC).date()
    first_promotion, first_state = after_hours._apply_promotion_consecutive_pass_gate(
        promotion=base_promotion,
        previous_state=None,
        run_date=first_day,
    )
    assert first_promotion["gate_passed"] is False
    assert first_promotion["status"] == "shadow"
    assert first_promotion["combined_gates"]["consecutive_passes"] is False
    assert first_promotion["consecutive_passes"]["count"] == 1
    assert first_state["count"] == 1

    second_day = datetime(2026, 1, 7, tzinfo=UTC).date()
    second_promotion, second_state = after_hours._apply_promotion_consecutive_pass_gate(
        promotion=base_promotion,
        previous_state={
            "promotion_consecutive_passes": first_state["count"],
            "promotion_last_pass_date": first_state["last_pass_date"],
        },
        run_date=second_day,
    )
    assert second_promotion["gate_passed"] is True
    assert second_promotion["status"] == "production"
    assert second_promotion["combined_gates"]["consecutive_passes"] is True
    assert second_promotion["consecutive_passes"]["count"] == 2
    assert second_state["count"] == 2


def test_promotion_consecutive_pass_gate_does_not_increment_twice_same_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_CONSECUTIVE_PASSES", "2")
    base_promotion = {
        "auto_promote": True,
        "gate_passed": True,
        "status": "production",
        "combined_gates": {"expectancy": True},
        "additional_gates": {},
    }
    run_day = datetime(2026, 1, 6, tzinfo=UTC).date()
    promotion, state = after_hours._apply_promotion_consecutive_pass_gate(
        promotion=base_promotion,
        previous_state={
            "promotion_consecutive_passes": 1,
            "promotion_last_pass_date": "2026-01-06",
        },
        run_date=run_day,
    )
    assert promotion["consecutive_passes"]["count"] == 1
    assert promotion["gate_passed"] is False
    assert state["count"] == 1


def test_after_hours_required_phase1_gate_blocks_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tickers = tmp_path / "tickers.csv"
    tickers.write_text("symbol\nAAPL\nMSFT\n", encoding="utf-8")
    tca_path = tmp_path / "tca_records.jsonl"
    _write_tca(tca_path, n=420)

    monkeypatch.setenv("AI_TRADING_TICKERS_FILE", str(tickers))
    monkeypatch.setenv("AI_TRADING_TCA_PATH", str(tca_path))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_REPORT_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MIN_ROWS", "120")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_CV_SPLITS", "3")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_LOOKBACK_DAYS", "420")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MIN_THRESHOLD_SUPPORT", "8")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RL_OVERLAY_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_AUTO_PROMOTE", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_POLICY", "strict")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_ROWS", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_SUPPORT", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_FOLDS", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_HIT_RATE", "0.0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_PROFITABLE_FOLDS", "0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_PROFITABLE_FOLD_RATIO", "0.0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_REQUIRE_PRIOR_IMPROVEMENT", "0")
    monkeypatch.setenv("AI_TRADING_POLICY_PROMOTION_MIN_OOS_SAMPLES", "1")
    monkeypatch.setenv("AI_TRADING_POLICY_PROMOTION_MIN_OOS_NET_BPS", "-9999")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_SENSITIVITY_SWEEP_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_EDGE_TARGET_EXPECTANCY_BPS", "-9999")
    monkeypatch.setenv("AI_TRADING_EDGE_TARGET_MAX_DRAWDOWN_BPS", "999999")
    monkeypatch.setenv("AI_TRADING_EDGE_TARGET_MAX_TURNOVER_RATIO", "1.0")
    monkeypatch.setenv("AI_TRADING_EDGE_TARGET_MIN_HIT_RATE_STABILITY", "0.0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_REQUIRE_PHASE1_GATE", "1")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE1_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE1_REQUIRE_PRIOR_IMPROVEMENT", "0")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE1_MIN_ROWS", "999999")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE1_MIN_SUPPORT", "1")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE1_MIN_EXPECTANCY_BPS", "-9999")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE1_MAX_DRAWDOWN_BPS", "999999")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE1_MAX_TURNOVER_RATIO", "1.0")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE1_MIN_HIT_RATE_STABILITY", "0.0")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE1_MAX_BRIER_SCORE", "1.0")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE1_MIN_PROFITABLE_FOLD_RATIO", "0.0")
    monkeypatch.setattr(
        after_hours,
        "_fetch_daily_bars",
        lambda symbol, _start, _end: _synthetic_daily(symbol),
    )

    result = after_hours.run_after_hours_training(
        now=datetime(2026, 1, 6, 21, 10, tzinfo=UTC),
    )

    assert result["status"] == "trained"
    assert result["governance_status"] == "shadow"
    assert result["promotion"]["strict_gates"]["rows"] is True
    assert result["roadmap"]["phase_1_week_1"]["required_for_promotion"] is True
    assert result["roadmap"]["phase_1_week_1"]["gates"]["rows"] is False
    assert result["promotion"]["additional_gates"]["phase1_week1"] is False
    assert result["promotion"]["combined_gates"]["phase1_week1"] is False
    assert result["promotion"]["gate_passed"] is False


def test_promotion_gate_bundle_can_ignore_sensitivity_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    best = after_hours.CandidateMetrics(
        name="xgboost",
        fold_count=5,
        profitable_fold_count=5,
        profitable_fold_ratio=1.0,
        support=120,
        mean_expectancy_bps=1.2,
        max_drawdown_bps=220.0,
        turnover_ratio=0.25,
        mean_hit_rate=0.5,
        hit_rate_stability=0.6,
        regime_metrics={},
        oof_probabilities=np.asarray([0.5, 0.6], dtype=float),
    )
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_AUTO_PROMOTE", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_POLICY", "strict")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_REQUIRE_SENSITIVITY", "0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_ROWS", "100")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_SUPPORT", "50")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_FOLDS", "4")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_HIT_RATE", "0.49")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_PROFITABLE_FOLDS", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_PROFITABLE_FOLD_RATIO", "0.0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_REQUIRE_PRIOR_IMPROVEMENT", "0")

    promotion = after_hours._promotion_gate_bundle(
        best=best,
        rows=400,
        edge_gates={
            "expectancy": True,
            "drawdown": True,
            "turnover": True,
            "stability": True,
            "sensitivity": False,
        },
    )

    assert promotion["require_sensitivity"] is False
    assert promotion["combined_gates"]["sensitivity"] is True
    assert promotion["gate_passed"] is True
    assert promotion["status"] == "production"


def test_promotion_confidence_gate_bundle_requires_effective_trades(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    best = after_hours.CandidateMetrics(
        name="xgboost",
        fold_count=5,
        profitable_fold_count=5,
        profitable_fold_ratio=1.0,
        support=24,
        mean_expectancy_bps=1.5,
        max_drawdown_bps=180.0,
        turnover_ratio=0.25,
        mean_hit_rate=0.70,
        hit_rate_stability=0.7,
        regime_metrics={},
        oof_probabilities=np.asarray([0.6, 0.7], dtype=float),
    )
    monkeypatch.setenv("AI_TRADING_PROMOTION_CONFIDENCE_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_PROMOTION_MIN_EFFECTIVE_TRADES", "100")
    monkeypatch.setenv("AI_TRADING_PROMOTION_WINRATE_CONFIDENCE_Z", "1.28")
    monkeypatch.setenv("AI_TRADING_PROMOTION_CAPTURE_CONFIDENCE_Z", "1.0")
    monkeypatch.setenv("AI_TRADING_PROMOTION_CAPTURE_BOOTSTRAP_ROUNDS", "200")

    gate = after_hours._promotion_confidence_gate_bundle(
        best=best,
        runtime_performance_gate={
            "execution_vs_alpha": {
                "execution_capture_ratio": 0.7,
                "daily": [
                    {
                        "date": "2026-01-01",
                        "expected_edge_per_accept_bps": 10.0,
                        "realized_net_edge_bps": 7.0,
                    },
                    {
                        "date": "2026-01-02",
                        "expected_edge_per_accept_bps": 9.0,
                        "realized_net_edge_bps": 6.0,
                    },
                ],
            }
        },
        live_execution_quality_gate={
            "thresholds": {"min_capture_ratio": 0.2, "max_slippage_drag_bps": 8.0}
        },
    )

    assert gate["enabled"] is True
    assert gate["gate_passed"] is False
    assert gate["reason"] == "insufficient_effective_trades"


def test_promotion_confidence_gate_bundle_passes_with_strong_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    best = after_hours.CandidateMetrics(
        name="xgboost",
        fold_count=5,
        profitable_fold_count=5,
        profitable_fold_ratio=1.0,
        support=320,
        mean_expectancy_bps=2.0,
        max_drawdown_bps=160.0,
        turnover_ratio=0.20,
        mean_hit_rate=0.66,
        hit_rate_stability=0.8,
        regime_metrics={},
        oof_probabilities=np.asarray([0.6, 0.7], dtype=float),
    )
    monkeypatch.setenv("AI_TRADING_PROMOTION_CONFIDENCE_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_PROMOTION_MIN_EFFECTIVE_TRADES", "120")
    monkeypatch.setenv("AI_TRADING_PROMOTION_WINRATE_CONFIDENCE_Z", "1.28")
    monkeypatch.setenv("AI_TRADING_PROMOTION_CAPTURE_CONFIDENCE_Z", "1.0")
    monkeypatch.setenv("AI_TRADING_PROMOTION_CAPTURE_BOOTSTRAP_ROUNDS", "300")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_HIT_RATE", "0.52")

    gate = after_hours._promotion_confidence_gate_bundle(
        best=best,
        runtime_performance_gate={
            "execution_vs_alpha": {
                "execution_capture_ratio": 0.72,
                "daily": [
                    {
                        "date": "2026-01-01",
                        "expected_edge_per_accept_bps": 10.0,
                        "realized_net_edge_bps": 7.0,
                    },
                    {
                        "date": "2026-01-02",
                        "expected_edge_per_accept_bps": 9.0,
                        "realized_net_edge_bps": 6.5,
                    },
                    {
                        "date": "2026-01-03",
                        "expected_edge_per_accept_bps": 11.0,
                        "realized_net_edge_bps": 8.0,
                    },
                    {
                        "date": "2026-01-04",
                        "expected_edge_per_accept_bps": 10.0,
                        "realized_net_edge_bps": 7.2,
                    },
                ],
            }
        },
        live_execution_quality_gate={
            "thresholds": {"min_capture_ratio": 0.35, "max_slippage_drag_bps": 4.5}
        },
    )

    assert gate["enabled"] is True
    assert gate["gate_passed"] is True
    assert gate["reason"] == "pass"
    assert gate["observed"]["win_rate_confidence_lower_bound"] is not None
    assert gate["observed"]["capture_ratio_confidence_lower_bound"] is not None


def test_promotion_confidence_gate_bundle_allows_missing_capture_data_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    best = after_hours.CandidateMetrics(
        name="xgboost",
        fold_count=5,
        profitable_fold_count=5,
        profitable_fold_ratio=1.0,
        support=420,
        mean_expectancy_bps=1.6,
        max_drawdown_bps=180.0,
        turnover_ratio=0.20,
        mean_hit_rate=0.64,
        hit_rate_stability=0.8,
        regime_metrics={},
        oof_probabilities=np.asarray([0.6, 0.7], dtype=float),
    )
    monkeypatch.setenv("AI_TRADING_PROMOTION_CONFIDENCE_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_PROMOTION_MIN_EFFECTIVE_TRADES", "120")
    monkeypatch.setenv("AI_TRADING_PROMOTION_WINRATE_CONFIDENCE_Z", "1.28")
    monkeypatch.setenv("AI_TRADING_PROMOTION_CAPTURE_CONFIDENCE_Z", "1.0")
    monkeypatch.setenv("AI_TRADING_PROMOTION_CAPTURE_CONFIDENCE_MIN_DAILY_SAMPLES", "5")
    monkeypatch.setenv("AI_TRADING_PROMOTION_CAPTURE_CONFIDENCE_REQUIRE_DATA", "0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_HIT_RATE", "0.52")

    gate = after_hours._promotion_confidence_gate_bundle(
        best=best,
        runtime_performance_gate={"execution_vs_alpha": {"execution_capture_ratio": 0.62, "daily": []}},
        live_execution_quality_gate={
            "thresholds": {"min_capture_ratio": 0.35, "max_slippage_drag_bps": 4.5}
        },
    )

    assert gate["enabled"] is True
    assert gate["gate_passed"] is True
    assert gate["reason"] == "pass"
    assert gate["observed"]["capture_ratio_daily_samples"] == 0
    assert gate["observed"]["capture_ratio_confidence_data_sufficient"] is False


def test_promotion_confidence_gate_bundle_can_require_capture_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    best = after_hours.CandidateMetrics(
        name="xgboost",
        fold_count=5,
        profitable_fold_count=5,
        profitable_fold_ratio=1.0,
        support=420,
        mean_expectancy_bps=1.6,
        max_drawdown_bps=180.0,
        turnover_ratio=0.20,
        mean_hit_rate=0.64,
        hit_rate_stability=0.8,
        regime_metrics={},
        oof_probabilities=np.asarray([0.6, 0.7], dtype=float),
    )
    monkeypatch.setenv("AI_TRADING_PROMOTION_CONFIDENCE_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_PROMOTION_MIN_EFFECTIVE_TRADES", "120")
    monkeypatch.setenv("AI_TRADING_PROMOTION_WINRATE_CONFIDENCE_Z", "1.28")
    monkeypatch.setenv("AI_TRADING_PROMOTION_CAPTURE_CONFIDENCE_Z", "1.0")
    monkeypatch.setenv("AI_TRADING_PROMOTION_CAPTURE_CONFIDENCE_MIN_DAILY_SAMPLES", "5")
    monkeypatch.setenv("AI_TRADING_PROMOTION_CAPTURE_CONFIDENCE_REQUIRE_DATA", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_HIT_RATE", "0.52")

    gate = after_hours._promotion_confidence_gate_bundle(
        best=best,
        runtime_performance_gate={"execution_vs_alpha": {"execution_capture_ratio": 0.62, "daily": []}},
        live_execution_quality_gate={
            "thresholds": {"min_capture_ratio": 0.35, "max_slippage_drag_bps": 4.5}
        },
    )

    assert gate["enabled"] is True
    assert gate["gate_passed"] is False
    assert gate["reason"] == "capture_confidence_data_unavailable"
    assert gate["observed"]["capture_ratio_confidence_data_sufficient"] is False


def test_promotion_confidence_gate_bundle_relaxes_hit_rate_when_runtime_data_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    best = after_hours.CandidateMetrics(
        name="xgboost",
        fold_count=5,
        profitable_fold_count=5,
        profitable_fold_ratio=1.0,
        support=940,
        mean_expectancy_bps=1.4,
        max_drawdown_bps=180.0,
        turnover_ratio=0.20,
        mean_hit_rate=(365.0 / 940.0),
        hit_rate_stability=0.8,
        regime_metrics={},
        oof_probabilities=np.asarray([0.6, 0.7], dtype=float),
    )
    monkeypatch.setenv("AI_TRADING_PROMOTION_CONFIDENCE_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_PROMOTION_MIN_EFFECTIVE_TRADES", "120")
    monkeypatch.setenv("AI_TRADING_PROMOTION_WINRATE_CONFIDENCE_Z", "1.28")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_HIT_RATE", "0.49")
    monkeypatch.setenv("AI_TRADING_PROMOTION_MIN_HIT_RATE_NO_RUNTIME_DATA", "0.35")
    monkeypatch.setenv(
        "AI_TRADING_PROMOTION_CONFIDENCE_RELAX_WHEN_RUNTIME_DATA_UNAVAILABLE",
        "1",
    )
    monkeypatch.setenv("AI_TRADING_PROMOTION_CAPTURE_CONFIDENCE_REQUIRE_DATA", "0")

    gate = after_hours._promotion_confidence_gate_bundle(
        best=best,
        runtime_performance_gate={
            "data_unavailable_fail_open_applied": True,
            "execution_vs_alpha": {"daily": []},
        },
        live_execution_quality_gate={
            "thresholds": {"min_capture_ratio": 0.10, "max_slippage_drag_bps": 9.0}
        },
    )

    assert gate["enabled"] is True
    assert gate["gate_passed"] is True
    assert gate["reason"] == "pass"
    assert gate["observed"]["runtime_data_unavailable"] is True
    assert gate["observed"]["win_rate_confidence_floor_effective"] == pytest.approx(0.35)


def test_threshold_by_regime_uses_drawdown_penalty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = pd.DataFrame(
        {
            "regime": ["regular"] * 12,
            "realized_edge_bps": [
                -18.82,
                21.9,
                29.6,
                -3.65,
                17.15,
                -12.05,
                14.79,
                -6.4,
                3.73,
                6.78,
                2.32,
                14.67,
            ],
        }
    )
    probabilities = np.asarray(
        [0.8, 0.8, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8, 0.8],
        dtype=float,
    )
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MIN_THRESHOLD_SUPPORT", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_THRESHOLD_MIN_EXPECTANCY_BPS", "0.0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_THRESHOLD_MAX_DRAWDOWN_BPS", "999999")

    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_THRESHOLD_DRAWDOWN_PENALTY", "0.0")
    no_penalty = after_hours._threshold_by_regime(
        dataset,
        probabilities,
        default_threshold=0.5,
    )
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_THRESHOLD_DRAWDOWN_PENALTY", "0.08")
    with_penalty = after_hours._threshold_by_regime(
        dataset,
        probabilities,
        default_threshold=0.5,
    )

    assert no_penalty["regular"] == pytest.approx(0.35)
    assert with_penalty["regular"] == pytest.approx(0.55)


def test_threshold_by_regime_respects_drawdown_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = pd.DataFrame(
        {
            "regime": ["regular"] * 10,
            "realized_edge_bps": [-30.0, 60.0, -25.0, 55.0, -20.0, 50.0, -10.0, 45.0, -5.0, 40.0],
        }
    )
    probabilities = np.asarray([0.8, 0.8, 0.75, 0.75, 0.7, 0.7, 0.55, 0.55, 0.5, 0.5], dtype=float)
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MIN_THRESHOLD_SUPPORT", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_THRESHOLD_DRAWDOWN_PENALTY", "0.0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_THRESHOLD_MIN_EXPECTANCY_BPS", "0.0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_THRESHOLD_MAX_DRAWDOWN_BPS", "15.0")

    thresholds = after_hours._threshold_by_regime(
        dataset,
        probabilities,
        default_threshold=0.5,
    )

    # This setup has no feasible threshold under the cap, so the selector should
    # fall back to the least-bad infeasible threshold (minimum drawdown first).
    assert thresholds["regular"] == pytest.approx(0.35)


def test_select_candidate_threshold_prefers_higher_quality_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = pd.DataFrame(
        {
            "regime": ["regular"] * 10,
            "realized_edge_bps": [5.0, 4.0, 3.0, 2.0, 1.0, -3.0, -2.0, -1.0, -4.0, -5.0],
        }
    )
    oof_probs = np.asarray([0.95, 0.9, 0.85, 0.8, 0.75, 0.6, 0.55, 0.5, 0.45, 0.4], dtype=float)
    fold_predictions = [(np.arange(10, dtype=int), oof_probs.copy())]
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_CANDIDATE_THRESHOLD_TUNING_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_CANDIDATE_THRESHOLD_GRID", "0.4,0.5,0.6")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MIN_THRESHOLD_SUPPORT", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_THRESHOLD_MIN_EXPECTANCY_BPS", "0.0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_THRESHOLD_MAX_DRAWDOWN_BPS", "999999")
    monkeypatch.setenv("AI_TRADING_EDGE_TARGET_MAX_TURNOVER_RATIO", "1.0")

    selected = after_hours._select_candidate_threshold(
        dataset=dataset,
        oof_probabilities=oof_probs,
        fold_predictions=fold_predictions,
        default_threshold=0.5,
    )

    assert selected["threshold"] == pytest.approx(0.6)
    assert selected["mean_expectancy_bps"] > 1.5
    assert selected["mean_hit_rate"] > 0.8


def test_select_candidate_threshold_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = pd.DataFrame(
        {
            "regime": ["regular"] * 10,
            "realized_edge_bps": [5.0, 4.0, 3.0, 2.0, 1.0, -3.0, -2.0, -1.0, -4.0, -5.0],
        }
    )
    oof_probs = np.asarray([0.95, 0.9, 0.85, 0.8, 0.75, 0.6, 0.55, 0.5, 0.45, 0.4], dtype=float)
    fold_predictions = [(np.arange(10, dtype=int), oof_probs.copy())]
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_CANDIDATE_THRESHOLD_TUNING_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_CANDIDATE_THRESHOLD_GRID", "0.4,0.6")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MIN_THRESHOLD_SUPPORT", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_THRESHOLD_MIN_EXPECTANCY_BPS", "0.0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_THRESHOLD_MAX_DRAWDOWN_BPS", "999999")
    monkeypatch.setenv("AI_TRADING_EDGE_TARGET_MAX_TURNOVER_RATIO", "1.0")

    selected = after_hours._select_candidate_threshold(
        dataset=dataset,
        oof_probabilities=oof_probs,
        fold_predictions=fold_predictions,
        default_threshold=0.5,
    )

    assert selected["threshold"] == pytest.approx(0.5)


def test_select_candidate_threshold_blocks_low_support_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = pd.DataFrame({"regime": ["regular", "regular"], "realized_edge_bps": [2.0, -1.0]})
    oof_probs = np.asarray([0.7, 0.3], dtype=float)
    fold_predictions = [(np.arange(2, dtype=int), oof_probs.copy())]

    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_CANDIDATE_THRESHOLD_REQUIRE_MIN_SUPPORT", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MIN_THRESHOLD_SUPPORT", "10")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_THRESHOLD_MIN_EXPECTANCY_BPS", "0.0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_THRESHOLD_MAX_DRAWDOWN_BPS", "999999")
    monkeypatch.setenv("AI_TRADING_EDGE_TARGET_MAX_TURNOVER_RATIO", "1.0")

    monkeypatch.setattr(after_hours, "_candidate_threshold_grid", lambda _default: [0.4, 0.6])

    def _fake_fold_metrics(
        *,
        edge: np.ndarray,
        regimes: np.ndarray,
        oof_probabilities: np.ndarray,
        fold_predictions: list[tuple[np.ndarray, np.ndarray]],
        threshold: float,
    ) -> dict[str, object]:
        del edge, regimes, oof_probabilities, fold_predictions
        if threshold < 0.5:
            return {
                "threshold": float(threshold),
                "support": 6,
                "mean_expectancy_bps": 6.0,
                "max_drawdown_bps": 200.0,
                "turnover_ratio": 0.3,
                "mean_hit_rate": 0.6,
                "hit_rate_stability": 0.8,
                "profitable_fold_count": 1,
                "profitable_fold_ratio": 1.0,
                "regime_metrics": {},
            }
        return {
            "threshold": float(threshold),
            "support": 3,
            "mean_expectancy_bps": 2.0,
            "max_drawdown_bps": 150.0,
            "turnover_ratio": 0.2,
            "mean_hit_rate": 0.55,
            "hit_rate_stability": 0.75,
            "profitable_fold_count": 1,
            "profitable_fold_ratio": 1.0,
            "regime_metrics": {},
        }

    monkeypatch.setattr(after_hours, "_fold_oof_threshold_metrics", _fake_fold_metrics)
    monkeypatch.setattr(
        after_hours,
        "_candidate_threshold_selection_score",
        lambda metrics: float(metrics["mean_expectancy_bps"]),
    )

    selected = after_hours._select_candidate_threshold(
        dataset=dataset,
        oof_probabilities=oof_probs,
        fold_predictions=fold_predictions,
        default_threshold=0.5,
    )

    assert selected == {}


def test_select_candidate_threshold_allows_low_support_fallback_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = pd.DataFrame({"regime": ["regular", "regular"], "realized_edge_bps": [2.0, -1.0]})
    oof_probs = np.asarray([0.7, 0.3], dtype=float)
    fold_predictions = [(np.arange(2, dtype=int), oof_probs.copy())]

    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_CANDIDATE_THRESHOLD_REQUIRE_MIN_SUPPORT", "0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MIN_THRESHOLD_SUPPORT", "10")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_THRESHOLD_MIN_EXPECTANCY_BPS", "0.0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_THRESHOLD_MAX_DRAWDOWN_BPS", "999999")
    monkeypatch.setenv("AI_TRADING_EDGE_TARGET_MAX_TURNOVER_RATIO", "1.0")

    monkeypatch.setattr(after_hours, "_candidate_threshold_grid", lambda _default: [0.4, 0.6])

    def _fake_fold_metrics(
        *,
        edge: np.ndarray,
        regimes: np.ndarray,
        oof_probabilities: np.ndarray,
        fold_predictions: list[tuple[np.ndarray, np.ndarray]],
        threshold: float,
    ) -> dict[str, object]:
        del edge, regimes, oof_probabilities, fold_predictions
        if threshold < 0.5:
            return {
                "threshold": float(threshold),
                "support": 6,
                "mean_expectancy_bps": 6.0,
                "max_drawdown_bps": 200.0,
                "turnover_ratio": 0.3,
                "mean_hit_rate": 0.6,
                "hit_rate_stability": 0.8,
                "profitable_fold_count": 1,
                "profitable_fold_ratio": 1.0,
                "regime_metrics": {},
            }
        return {
            "threshold": float(threshold),
            "support": 3,
            "mean_expectancy_bps": 2.0,
            "max_drawdown_bps": 150.0,
            "turnover_ratio": 0.2,
            "mean_hit_rate": 0.55,
            "hit_rate_stability": 0.75,
            "profitable_fold_count": 1,
            "profitable_fold_ratio": 1.0,
            "regime_metrics": {},
        }

    monkeypatch.setattr(after_hours, "_fold_oof_threshold_metrics", _fake_fold_metrics)
    monkeypatch.setattr(
        after_hours,
        "_candidate_threshold_selection_score",
        lambda metrics: float(metrics["mean_expectancy_bps"]),
    )

    selected = after_hours._select_candidate_threshold(
        dataset=dataset,
        oof_probabilities=oof_probs,
        fold_predictions=fold_predictions,
        default_threshold=0.5,
    )

    assert selected["threshold"] == pytest.approx(0.4)
    assert selected["support"] == 6


def test_promotion_gate_bundle_requires_profitable_folds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    best = after_hours.CandidateMetrics(
        name="xgboost",
        fold_count=5,
        profitable_fold_count=2,
        profitable_fold_ratio=0.4,
        support=120,
        mean_expectancy_bps=1.2,
        max_drawdown_bps=220.0,
        turnover_ratio=0.25,
        mean_hit_rate=0.5,
        hit_rate_stability=0.6,
        regime_metrics={},
        oof_probabilities=np.asarray([0.5, 0.6], dtype=float),
    )
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_AUTO_PROMOTE", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_POLICY", "strict")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_ROWS", "100")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_SUPPORT", "50")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_FOLDS", "4")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_HIT_RATE", "0.49")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_PROFITABLE_FOLDS", "3")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_PROFITABLE_FOLD_RATIO", "0.5")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_REQUIRE_PRIOR_IMPROVEMENT", "0")

    promotion = after_hours._promotion_gate_bundle(
        best=best,
        rows=400,
        edge_gates={
            "expectancy": True,
            "drawdown": True,
            "turnover": True,
            "stability": True,
            "sensitivity": True,
        },
    )

    assert promotion["strict_gates"]["profitable_folds"] is False
    assert promotion["strict_gates"]["profitable_fold_ratio"] is False
    assert promotion["gate_passed"] is False
    assert promotion["status"] == "shadow"


def test_promotion_gate_bundle_requires_prior_model_improvement_margin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    best = after_hours.CandidateMetrics(
        name="xgboost",
        fold_count=5,
        profitable_fold_count=5,
        profitable_fold_ratio=1.0,
        support=120,
        mean_expectancy_bps=1.8,
        max_drawdown_bps=250.0,
        turnover_ratio=0.25,
        mean_hit_rate=0.52,
        hit_rate_stability=0.6,
        regime_metrics={},
        oof_probabilities=np.asarray([0.5, 0.6], dtype=float),
    )
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_AUTO_PROMOTE", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_POLICY", "strict")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_ROWS", "100")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_SUPPORT", "50")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_FOLDS", "4")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_HIT_RATE", "0.49")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_PROFITABLE_FOLDS", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_PROFITABLE_FOLD_RATIO", "0.0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_REQUIRE_PRIOR_IMPROVEMENT", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_DRAWDOWN_PENALTY", "0.003")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_SCORE_MARGIN", "0.6")

    promotion = after_hours._promotion_gate_bundle(
        best=best,
        rows=400,
        edge_gates={
            "expectancy": True,
            "drawdown": True,
            "turnover": True,
            "stability": True,
            "sensitivity": True,
        },
        prior_metrics={
            "mean_expectancy_bps": 2.0,
            "max_drawdown_bps": 240.0,
            "model_id": "prior-model",
            "report_path": "/tmp/prior-report.json",
            "governance_status": "production",
        },
    )

    assert promotion["strict_gates"]["prior_model_improvement"] is False
    assert promotion["prior_model_comparison"]["available"] is True
    assert promotion["prior_model_comparison"]["gate"] is False
    assert promotion["gate_passed"] is False
    assert promotion["status"] == "shadow"


def test_champion_challenger_ab_gate_passes_on_meaningful_uplift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    best = after_hours.CandidateMetrics(
        name="xgboost",
        fold_count=5,
        profitable_fold_count=5,
        profitable_fold_ratio=1.0,
        support=120,
        mean_expectancy_bps=2.0,
        max_drawdown_bps=220.0,
        turnover_ratio=0.25,
        mean_hit_rate=0.55,
        hit_rate_stability=0.7,
        regime_metrics={},
        oof_probabilities=np.asarray([0.5, 0.6], dtype=float),
        fold_expectancy_bps=(2.6, 2.3, 2.5, 2.4, 2.7),
    )
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_AB_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_AB_REQUIRE_SIGNIFICANCE", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_AB_MIN_FOLDS", "4")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_AB_MIN_DELTA_BPS", "0.8")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_AB_MAX_P_VALUE", "0.25")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_AB_REQUIRE_MATCHING_RUNTIME_ASSUMPTIONS", "1")

    bundle = after_hours._champion_challenger_ab_gate_bundle(
        best=best,
        prior_metrics={
            "fold_expectancy_bps": [0.8, 1.0, 0.9, 1.1, 0.7],
            "runtime_performance_thresholds": {
                "trade_fill_source": "all",
                "lookback_days": 5,
                "min_used_days": 3,
                "min_closed_trades": 50,
                "min_profit_factor": 0.9,
                "min_win_rate": 0.5,
                "min_net_pnl": -500.0,
                "min_acceptance_rate": 0.02,
                "min_expected_net_edge_bps": -50.0,
            },
        },
        runtime_performance_gate={
            "thresholds": {
                "trade_fill_source": "all",
                "lookback_days": 5,
                "min_used_days": 3,
                "min_closed_trades": 50,
                "min_profit_factor": 0.9,
                "min_win_rate": 0.5,
                "min_net_pnl": -500.0,
                "min_acceptance_rate": 0.02,
                "min_expected_net_edge_bps": -50.0,
            }
        },
    )

    assert bundle["enabled"] is True
    assert bundle["required_for_promotion"] is True
    assert bundle["gate_passed"] is True
    assert bundle["reason"] == "stat_sig_pass"
    assert bundle["observed"]["delta_expectancy_bps"] > 0.8
    assert bundle["observed"]["p_value"] <= 0.25


def test_champion_challenger_ab_gate_blocks_assumption_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    best = after_hours.CandidateMetrics(
        name="logreg",
        fold_count=5,
        profitable_fold_count=4,
        profitable_fold_ratio=0.8,
        support=120,
        mean_expectancy_bps=1.4,
        max_drawdown_bps=220.0,
        turnover_ratio=0.25,
        mean_hit_rate=0.55,
        hit_rate_stability=0.7,
        regime_metrics={},
        oof_probabilities=np.asarray([0.5, 0.6], dtype=float),
        fold_expectancy_bps=(1.6, 1.5, 1.4, 1.5, 1.6),
    )
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_AB_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_AB_REQUIRE_SIGNIFICANCE", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_AB_REQUIRE_MATCHING_RUNTIME_ASSUMPTIONS", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_AB_ALLOW_MISSING_PRIOR", "0")

    bundle = after_hours._champion_challenger_ab_gate_bundle(
        best=best,
        prior_metrics={
            "fold_expectancy_bps": [0.9, 1.0, 1.1, 0.8, 1.0],
            "runtime_performance_thresholds": {
                "trade_fill_source": "live",
                "lookback_days": 5,
            },
        },
        runtime_performance_gate={
            "thresholds": {
                "trade_fill_source": "all",
                "lookback_days": 5,
            }
        },
    )

    assert bundle["enabled"] is True
    assert bundle["required_for_promotion"] is True
    assert bundle["gate_passed"] is False
    assert bundle["reason"] == "runtime_assumptions_mismatch"
    assert bundle["observed"]["runtime_assumptions_match"] is False
    assert "trade_fill_source" in bundle["observed"]["runtime_assumptions_mismatch"]


def test_champion_challenger_ab_gate_is_advisory_for_shadow_prior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    best = after_hours.CandidateMetrics(
        name="logreg",
        fold_count=5,
        profitable_fold_count=4,
        profitable_fold_ratio=0.8,
        support=120,
        mean_expectancy_bps=1.4,
        max_drawdown_bps=220.0,
        turnover_ratio=0.25,
        mean_hit_rate=0.55,
        hit_rate_stability=0.7,
        regime_metrics={},
        oof_probabilities=np.asarray([0.5, 0.6], dtype=float),
        fold_expectancy_bps=(1.6, 1.5, 1.4, 1.5, 1.6),
    )
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_AB_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_AB_REQUIRE_SIGNIFICANCE", "1")
    monkeypatch.setenv(
        "AI_TRADING_AFTER_HOURS_PROMOTION_AB_ADVISORY_FOR_NON_PRODUCTION_PRIOR",
        "1",
    )

    bundle = after_hours._champion_challenger_ab_gate_bundle(
        best=best,
        prior_metrics={
            "governance_status": "shadow",
            "fold_expectancy_bps": [1.4, 1.4, 1.5, 1.5, 1.4],
            "runtime_performance_thresholds": {
                "trade_fill_source": "live",
                "lookback_days": 5,
            },
        },
        runtime_performance_gate={
            "thresholds": {
                "trade_fill_source": "live",
                "lookback_days": 5,
            }
        },
    )

    assert bundle["enabled"] is True
    assert bundle["required_for_promotion"] is False
    assert bundle["gate_passed"] is False
    assert bundle["reason"] == "stat_sig_not_met"


def test_phase1_week1_gate_bundle_reports_blockers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    best = after_hours.CandidateMetrics(
        name="logreg",
        fold_count=5,
        profitable_fold_count=3,
        profitable_fold_ratio=0.6,
        support=180,
        mean_expectancy_bps=2.1,
        max_drawdown_bps=320.0,
        turnover_ratio=0.2,
        mean_hit_rate=0.53,
        hit_rate_stability=0.7,
        regime_metrics={},
        oof_probabilities=np.asarray([0.45, 0.62], dtype=float),
        brier_score=0.31,
    )
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE1_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE1_REQUIRE_PRIOR_IMPROVEMENT", "0")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE1_MIN_ROWS", "100")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE1_MIN_SUPPORT", "100")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE1_MIN_EXPECTANCY_BPS", "1.0")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE1_MAX_DRAWDOWN_BPS", "9999")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE1_MAX_TURNOVER_RATIO", "0.5")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE1_MIN_HIT_RATE_STABILITY", "0.5")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE1_MAX_BRIER_SCORE", "0.2")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE1_MIN_PROFITABLE_FOLD_RATIO", "0.4")

    phase1 = after_hours._phase1_week1_gate_bundle(
        best=best,
        rows=400,
        sensitivity_sweep={"gate": False},
        prior_metrics=None,
    )

    assert phase1["enabled"] is True
    assert phase1["gates"]["brier"] is False
    assert phase1["gates"]["sensitivity"] is False
    assert phase1["gate_passed"] is False


def test_phase1_week1_gate_blocks_material_regime_calibration_degradation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    best = after_hours.CandidateMetrics(
        name="logreg",
        fold_count=5,
        profitable_fold_count=4,
        profitable_fold_ratio=0.8,
        support=200,
        mean_expectancy_bps=3.0,
        max_drawdown_bps=250.0,
        turnover_ratio=0.2,
        mean_hit_rate=0.55,
        hit_rate_stability=0.75,
        regime_metrics={},
        oof_probabilities=np.asarray([0.45, 0.62], dtype=float),
        brier_score=0.2,
        regime_calibration={
            "uptrend": {"support": 120.0, "brier_score": 0.26, "ece": 0.12},
            "volatile": {"support": 80.0, "brier_score": 0.23, "ece": 0.10},
        },
    )
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE1_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE1_REQUIRE_PRIOR_IMPROVEMENT", "0")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE1_MIN_ROWS", "100")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE1_MIN_SUPPORT", "100")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE1_MIN_EXPECTANCY_BPS", "1.0")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE1_MAX_DRAWDOWN_BPS", "9999")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE1_MAX_TURNOVER_RATIO", "0.5")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE1_MIN_HIT_RATE_STABILITY", "0.5")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE1_MAX_BRIER_SCORE", "0.5")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE1_MIN_PROFITABLE_FOLD_RATIO", "0.4")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE1_REGIME_CALIBRATION_GATE_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE1_MAX_REGIME_BRIER_DELTA", "0.02")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE1_MAX_REGIME_ECE_DELTA", "0.03")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE1_MAX_DEGRADED_REGIMES", "0")

    phase1 = after_hours._phase1_week1_gate_bundle(
        best=best,
        rows=400,
        sensitivity_sweep={"gate": True},
        prior_metrics={
            "regime_calibration": {
                "uptrend": {"support": 120.0, "brier_score": 0.20, "ece": 0.06},
                "volatile": {"support": 80.0, "brier_score": 0.21, "ece": 0.08},
            }
        },
        regime_calibration=best.regime_calibration,
    )

    assert phase1["calibration_diagnostics"]["degraded_regimes"] >= 1
    assert phase1["gates"]["regime_calibration"] is False
    assert phase1["gate_passed"] is False


def test_promotion_gate_bundle_honors_additional_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    best = after_hours.CandidateMetrics(
        name="xgboost",
        fold_count=5,
        profitable_fold_count=5,
        profitable_fold_ratio=1.0,
        support=120,
        mean_expectancy_bps=2.0,
        max_drawdown_bps=220.0,
        turnover_ratio=0.25,
        mean_hit_rate=0.55,
        hit_rate_stability=0.7,
        regime_metrics={},
        oof_probabilities=np.asarray([0.5, 0.6], dtype=float),
        brier_score=0.1,
    )
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_AUTO_PROMOTE", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_POLICY", "strict")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_ROWS", "100")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_SUPPORT", "50")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_FOLDS", "4")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_HIT_RATE", "0.49")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_PROFITABLE_FOLDS", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_PROFITABLE_FOLD_RATIO", "0.0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTION_REQUIRE_PRIOR_IMPROVEMENT", "0")

    promotion = after_hours._promotion_gate_bundle(
        best=best,
        rows=400,
        edge_gates={
            "expectancy": True,
            "drawdown": True,
            "turnover": True,
            "stability": True,
            "sensitivity": True,
        },
        additional_gates={"phase1_week1": False},
    )

    assert promotion["additional_gates"]["phase1_week1"] is False
    assert promotion["combined_gates"]["phase1_week1"] is False
    assert promotion["gate_passed"] is False
    assert promotion["status"] == "shadow"


def test_on_market_close_runs_after_hours_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ai_trading.core import bot_engine

    calls: dict[str, int] = {"after_hours": 0, "legacy": 0}

    class _FixedDateTime:
        @staticmethod
        def now(_tz=None):
            return datetime(2026, 1, 6, 21, 10, tzinfo=UTC)

    marker_path = tmp_path / "after_hours.marker.json"
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_ONCE_PER_DAY", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_MARKER_PATH", str(marker_path))
    monkeypatch.setenv("AI_TRADING_LEGACY_DAILY_RETRAIN_ENABLED", "1")
    monkeypatch.setattr(bot_engine, "dt_", _FixedDateTime)
    monkeypatch.setattr(bot_engine, "market_is_open", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(bot_engine, "get_ctx", lambda: object())
    monkeypatch.setattr(
        bot_engine,
        "load_or_retrain_daily",
        lambda *_args, **_kwargs: calls.__setitem__("legacy", calls["legacy"] + 1),
    )
    def _run_after_hours_training(**_kwargs):
        calls["after_hours"] += 1
        return {"status": "trained", "model_id": "m-1", "model_name": "logreg"}

    monkeypatch.setattr(
        after_hours,
        "run_after_hours_training",
        _run_after_hours_training,
    )

    bot_engine.on_market_close()
    assert calls["after_hours"] == 1
    assert calls["legacy"] == 1


def test_on_market_close_skips_after_hours_pipeline_when_marker_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ai_trading.core import bot_engine

    class _FixedDateTime:
        @staticmethod
        def now(_tz=None):
            return datetime(2026, 1, 6, 21, 10, tzinfo=UTC)

    marker_path = tmp_path / "after_hours.marker.json"
    marker_path.write_text(
        json.dumps({"date": "2026-01-06", "status": "trained"}) + "\n",
        encoding="utf-8",
    )
    calls: dict[str, int] = {"after_hours": 0}
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_ONCE_PER_DAY", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_MARKER_PATH", str(marker_path))
    monkeypatch.setenv("AI_TRADING_LEGACY_DAILY_RETRAIN_ENABLED", "0")
    monkeypatch.setattr(bot_engine, "dt_", _FixedDateTime)
    monkeypatch.setattr(bot_engine, "market_is_open", lambda *_args, **_kwargs: False)
    def _run_after_hours_training(**_kwargs):
        calls["after_hours"] += 1
        return {"status": "trained"}

    monkeypatch.setattr(
        after_hours,
        "run_after_hours_training",
        _run_after_hours_training,
    )

    bot_engine.on_market_close()

    assert calls["after_hours"] == 0


def test_on_market_close_writes_after_hours_training_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ai_trading.core import bot_engine

    class _FixedDateTime:
        @staticmethod
        def now(_tz=None):
            return datetime(2026, 1, 6, 21, 10, tzinfo=UTC)

    marker_path = tmp_path / "after_hours.marker.json"
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_ONCE_PER_DAY", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_MARKER_PATH", str(marker_path))
    monkeypatch.setenv("AI_TRADING_LEGACY_DAILY_RETRAIN_ENABLED", "0")
    monkeypatch.setattr(bot_engine, "dt_", _FixedDateTime)
    monkeypatch.setattr(bot_engine, "market_is_open", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        after_hours,
        "run_after_hours_training",
        lambda **_kwargs: {
            "status": "trained",
            "model_id": "m-1",
            "model_name": "logreg",
            "governance_status": "production",
        },
    )

    bot_engine.on_market_close()

    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    assert payload["date"] == "2026-01-06"
    assert payload["status"] == "trained"
    assert payload["model_id"] == "m-1"
    assert payload["model_name"] == "logreg"


def test_marker_path_resolution_prefers_data_dir_for_relative_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ai_trading.core import bot_engine

    data_dir = tmp_path / "state_root"
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_MARKER_PATH", "runtime/custom.marker.json")
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(data_dir))
    monkeypatch.delenv("STATE_DIRECTORY", raising=False)

    resolved = bot_engine._resolve_after_hours_training_marker_path()
    assert resolved == (data_dir / "runtime/custom.marker.json").resolve()


def test_marker_write_falls_back_when_requested_path_unwritable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ai_trading.core import bot_engine

    requested_marker = tmp_path / "readonly_root" / "after_hours.marker.json"
    fallback_data_dir = tmp_path / "fallback_data"
    monkeypatch.setenv(
        "AI_TRADING_AFTER_HOURS_TRAINING_MARKER_PATH",
        str(requested_marker),
    )
    monkeypatch.setattr(bot_engine.paths, "DATA_DIR", fallback_data_dir)

    original_writer = bot_engine._write_after_hours_training_marker_file

    def _writer_with_primary_failure(path: Path, payload) -> None:
        if path == requested_marker:
            raise PermissionError("read-only marker path")
        original_writer(path, payload)

    monkeypatch.setattr(
        bot_engine,
        "_write_after_hours_training_marker_file",
        _writer_with_primary_failure,
    )

    bot_engine._write_after_hours_training_marker(
        "2026-01-06",
        {
            "status": "trained",
            "model_id": "m-1",
            "model_name": "logreg",
            "governance_status": "production",
        },
    )

    fallback_marker = (fallback_data_dir / "runtime/after_hours_training.marker.json").resolve()
    assert fallback_marker.exists()
    payload = json.loads(fallback_marker.read_text(encoding="utf-8"))
    assert payload["date"] == "2026-01-06"
    assert payload["status"] == "trained"
    assert bot_engine._after_hours_training_completed_for_date("2026-01-06")


def test_on_market_close_overnight_catchup_writes_previous_business_marker_date(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ai_trading.core import bot_engine

    class _FixedDateTime:
        @staticmethod
        def now(_tz=None):
            return datetime(2026, 1, 7, 5, 10, tzinfo=UTC)  # 00:10 New York

    marker_path = tmp_path / "after_hours.marker.json"
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_ONCE_PER_DAY", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_CATCHUP_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_MARKER_PATH", str(marker_path))
    monkeypatch.setenv("AI_TRADING_LEGACY_DAILY_RETRAIN_ENABLED", "0")
    monkeypatch.setattr(bot_engine, "dt_", _FixedDateTime)
    monkeypatch.setattr(bot_engine, "market_is_open", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        after_hours,
        "run_after_hours_training",
        lambda **_kwargs: {
            "status": "trained",
            "model_id": "m-overnight",
            "model_name": "logreg",
            "governance_status": "production",
        },
    )

    bot_engine.on_market_close()

    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    assert payload["date"] == "2026-01-06"
    assert payload["status"] == "trained"


def test_after_hours_training_handles_leakage_assertions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tickers = tmp_path / "tickers.csv"
    tickers.write_text("symbol\nAAPL\nMSFT\n", encoding="utf-8")
    tca_path = tmp_path / "tca_records.jsonl"
    _write_tca(tca_path, n=420)

    monkeypatch.setenv("AI_TRADING_TICKERS_FILE", str(tickers))
    monkeypatch.setenv("AI_TRADING_TCA_PATH", str(tca_path))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_REPORT_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MIN_ROWS", "120")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_CV_SPLITS", "3")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_LOOKBACK_DAYS", "420")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MIN_THRESHOLD_SUPPORT", "8")
    monkeypatch.setattr(
        after_hours,
        "_fetch_daily_bars",
        lambda symbol, _start, _end: _synthetic_daily(symbol),
    )
    monkeypatch.setattr(
        after_hours,
        "run_leakage_guards",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("forced leakage")),
    )

    result = after_hours.run_after_hours_training(
        now=datetime(2026, 1, 6, 21, 10, tzinfo=UTC),  # 16:10 New York
    )
    assert result["status"] == "skipped"
    assert result["reason"] == "no_candidate_models"


def test_after_hours_training_no_global_leakage_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    tickers = tmp_path / "tickers.csv"
    tickers.write_text("symbol\nAAPL\nMSFT\n", encoding="utf-8")
    tca_path = tmp_path / "tca_records.jsonl"
    _write_tca(tca_path, n=420)

    monkeypatch.setenv("AI_TRADING_TICKERS_FILE", str(tickers))
    monkeypatch.setenv("AI_TRADING_TCA_PATH", str(tca_path))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_REPORT_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MIN_ROWS", "120")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_CV_SPLITS", "3")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_LOOKBACK_DAYS", "420")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MIN_THRESHOLD_SUPPORT", "8")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RL_OVERLAY_ENABLED", "0")
    monkeypatch.setattr(
        after_hours,
        "_fetch_daily_bars",
        lambda symbol, _start, _end: _synthetic_daily(symbol),
    )
    caplog.set_level("WARNING")

    result = after_hours.run_after_hours_training(
        now=datetime(2026, 1, 6, 21, 10, tzinfo=UTC),  # 16:10 New York
    )
    assert result["status"] == "trained"
    assert not any(
        "AFTER_HOURS_GLOBAL_LEAKAGE_GUARD_WARNING" in record.message
        for record in caplog.records
    )


def test_after_hours_uses_dedicated_ticker_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live_tickers = tmp_path / "tickers.live.csv"
    live_tickers.write_text("symbol\nAAPL\nMSFT\n", encoding="utf-8")
    train_tickers = tmp_path / "tickers.train.csv"
    train_tickers.write_text("symbol\nSPY\nQQQ\n", encoding="utf-8")
    tca_path = tmp_path / "tca_records.jsonl"
    _write_tca(tca_path, n=420)

    monkeypatch.setenv("AI_TRADING_TICKERS_FILE", str(live_tickers))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TICKERS_CSV", str(train_tickers))
    monkeypatch.setenv("AI_TRADING_TCA_PATH", str(tca_path))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_REPORT_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MIN_ROWS", "120")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_CV_SPLITS", "3")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_LOOKBACK_DAYS", "420")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MIN_THRESHOLD_SUPPORT", "8")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RL_OVERLAY_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_LIGHTGBM_VERBOSITY", "-1")
    monkeypatch.setattr(
        after_hours,
        "_fetch_daily_bars",
        lambda symbol, _start, _end: _synthetic_daily(symbol),
    )

    result = after_hours.run_after_hours_training(
        now=datetime(2026, 1, 6, 21, 10, tzinfo=UTC),  # 16:10 New York
    )
    assert result["status"] == "trained"
    assert set(json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))["symbols"]) == {
        "QQQ",
        "SPY",
    }


def test_cost_floor_estimate_stabilizes_with_tca_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(UTC)
    old = now - timedelta(days=200)
    records = [
        {"ts": now.isoformat(), "status": "filled", "is_bps": 8.0},
        {"ts": now.isoformat(), "status": "filled", "is_bps": 10.0},
        {"ts": now.isoformat(), "status": "partially_filled", "is_bps": 12.0},
        {"ts": now.isoformat(), "status": "filled", "is_bps": 14.0},
        {"ts": now.isoformat(), "status": "filled", "is_bps": 500.0},  # outlier
        {"ts": old.isoformat(), "status": "filled", "is_bps": 60.0},  # stale
        {"ts": now.isoformat(), "status": "new", "is_bps": 2.0},  # not filled
    ]
    monkeypatch.setenv("AI_TRADING_COST_FLOOR_BPS", "12")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_COST_FLOOR_MIN_BPS", "4")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_COST_FLOOR_MAX_BPS", "25")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_COST_FLOOR_LOOKBACK_DAYS", "45")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_COST_FLOOR_MIN_SAMPLES", "3")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_COST_FLOOR_REQUIRE_FILLED", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_COST_FLOOR_OUTLIER_BPS", "120")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_COST_FLOOR_QUANTILE", "0.5")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_COST_FLOOR_TCA_WEIGHT", "1.0")

    value = after_hours._estimate_cost_floor_bps(records)
    assert value == pytest.approx(11.0, abs=0.05)


def test_cost_floor_estimate_falls_back_to_baseline_when_samples_low(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(UTC)
    records = [
        {"ts": now.isoformat(), "status": "filled", "is_bps": 6.0},
        {"ts": now.isoformat(), "status": "filled", "is_bps": 9.0},
    ]
    monkeypatch.setenv("AI_TRADING_COST_FLOOR_BPS", "12")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_COST_FLOOR_MIN_BPS", "4")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_COST_FLOOR_MAX_BPS", "25")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_COST_FLOOR_MIN_SAMPLES", "10")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_COST_FLOOR_REQUIRE_FILLED", "1")

    value = after_hours._estimate_cost_floor_bps(records)
    assert value == pytest.approx(12.0, abs=0.001)


def test_maybe_train_rl_overlay_passes_price_series_and_registry_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ai_trading.rl_trading.train as train_mod

    n_rows = 140
    dataset = pd.DataFrame(
        {
            "rsi": np.linspace(45.0, 65.0, n_rows),
            "macd": np.linspace(-0.2, 0.3, n_rows),
            "atr": np.linspace(1.0, 2.5, n_rows),
            "vwap": np.linspace(99.0, 105.0, n_rows),
            "sma_50": np.linspace(98.0, 104.0, n_rows),
            "sma_200": np.linspace(95.0, 101.0, n_rows),
            "close": np.linspace(100.0, 110.0, n_rows),
        }
    )
    captured: dict[str, object] = {}

    class DummyTrainer:
        def __init__(
            self,
            *,
            algorithm: str,
            total_timesteps: int,
            eval_freq: int,
            early_stopping_patience: int,
            seed: int,
        ) -> None:
            captured["init"] = {
                "algorithm": algorithm,
                "total_timesteps": total_timesteps,
                "eval_freq": eval_freq,
                "early_stopping_patience": early_stopping_patience,
                "seed": seed,
            }

        def train(
            self,
            *,
            data: np.ndarray,
            env_params: dict[str, object],
            save_path: str,
        ) -> dict[str, object]:
            captured["train_data_shape"] = tuple(np.asarray(data).shape)
            captured["env_params"] = dict(env_params)
            captured["save_path"] = save_path
            return {
                "final_evaluation": {"mean_reward": 1.25},
                "model_id": "rl-model-123",
                "governance_status": "production",
            }

    def fake_train_multi_seed(
        *,
        data: np.ndarray,
        seeds: list[int],
        algorithm: str,
        total_timesteps: int,
        eval_freq: int,
        early_stopping_patience: int,
        env_params: dict[str, object],
        model_params: dict[str, object] | None,
        save_root: str | None,
    ) -> dict[str, object]:
        captured["multi_seed"] = {
            "data_shape": tuple(np.asarray(data).shape),
            "seeds": list(seeds),
            "algorithm": algorithm,
            "total_timesteps": total_timesteps,
            "eval_freq": eval_freq,
            "early_stopping_patience": early_stopping_patience,
            "env_params": dict(env_params),
            "model_params": model_params,
            "save_root": save_root,
        }
        return {"run_count": len(seeds), "seeds": list(seeds)}

    monkeypatch.setattr(train_mod, "RLTrainer", DummyTrainer)
    monkeypatch.setattr(train_mod, "train_multi_seed", fake_train_multi_seed)
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RL_OVERLAY_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RL_TIMESTEPS", "2500")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RL_ALGO", "PPO")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RL_DIR", str(tmp_path / "rl"))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RL_MULTI_SEED_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RL_MULTI_SEEDS", "7,11,7,invalid")
    monkeypatch.setenv("AI_TRADING_RL_MIN_MEAN_REWARD", "0.0")
    monkeypatch.setenv("AI_TRADING_RL_REQUIRE_BASELINE_EXPECTANCY_BPS", "1.0")
    monkeypatch.setenv("AI_TRADING_SEED", "42")

    result = after_hours._maybe_train_rl_overlay(
        dataset,
        now_utc=datetime(2026, 1, 6, 21, 10, tzinfo=UTC),
        baseline_expectancy_bps=2.2,
        dataset_fingerprint="dataset-fp-001",
        feature_hash="feature-hash-001",
        governance_status="production",
    )

    assert result["enabled"] is True
    assert result["trained"] is True
    assert result["recommend_use_rl_agent"] is True
    assert result["model_id"] == "rl-model-123"
    assert result["governance_status"] == "production"
    assert result["multi_seed_summary"] == {"run_count": 2, "seeds": [7, 11]}

    env_params = captured["env_params"]
    assert isinstance(env_params, dict)
    assert env_params["dataset_fingerprint"] == "dataset-fp-001"
    assert env_params["feature_spec_hash"] == "feature-hash-001"
    assert env_params["registry_strategy"] == "rl_overlay"
    assert env_params["registry_model_type"] == "ppo"
    assert env_params["registry_requested_status"] == "production"
    np.testing.assert_allclose(
        np.asarray(env_params["price_series"], dtype=float),
        dataset["close"].to_numpy(dtype=float),
    )

    multi_seed = captured["multi_seed"]
    assert isinstance(multi_seed, dict)
    assert multi_seed["seeds"] == [7, 11]
    multi_env_params = multi_seed["env_params"]
    assert isinstance(multi_env_params, dict)
    assert multi_env_params["dataset_fingerprint"] == "dataset-fp-001"
    assert multi_env_params["feature_spec_hash"] == "feature-hash-001"
    np.testing.assert_allclose(
        np.asarray(multi_env_params["price_series"], dtype=float),
        dataset["close"].to_numpy(dtype=float),
    )


def test_maybe_train_rl_overlay_promotes_runtime_rl_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ai_trading.rl_trading.train as train_mod

    n_rows = 140
    dataset = pd.DataFrame(
        {
            "rsi": np.linspace(45.0, 65.0, n_rows),
            "macd": np.linspace(-0.2, 0.3, n_rows),
            "atr": np.linspace(1.0, 2.5, n_rows),
            "vwap": np.linspace(99.0, 105.0, n_rows),
            "sma_50": np.linspace(98.0, 104.0, n_rows),
            "sma_200": np.linspace(95.0, 101.0, n_rows),
            "close": np.linspace(100.0, 110.0, n_rows),
        }
    )

    class DummyTrainer:
        def __init__(
            self,
            *,
            algorithm: str,
            total_timesteps: int,
            eval_freq: int,
            early_stopping_patience: int,
            seed: int,
        ) -> None:
            self.algorithm = algorithm

        def train(
            self,
            *,
            data: np.ndarray,
            env_params: dict[str, object],
            save_path: str,
        ) -> dict[str, object]:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            artifact = save_dir / f"model_{self.algorithm.lower()}.zip"
            artifact.write_bytes(b"rl-model")
            return {
                "final_evaluation": {"mean_reward": 1.0},
                "model_id": "rl-model-promote",
                "governance_status": "production",
            }

    monkeypatch.setattr(train_mod, "RLTrainer", DummyTrainer)
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RL_OVERLAY_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RL_TIMESTEPS", "2500")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RL_ALGO", "PPO")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RL_DIR", str(tmp_path / "rl"))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTE_RL_PATH", "1")
    monkeypatch.setenv("AI_TRADING_RL_MODEL_PATH", str(tmp_path / "runtime_rl.zip"))
    monkeypatch.setenv("AI_TRADING_RL_MIN_MEAN_REWARD", "0.0")
    monkeypatch.setenv("AI_TRADING_RL_REQUIRE_BASELINE_EXPECTANCY_BPS", "0.0")

    result = after_hours._maybe_train_rl_overlay(
        dataset,
        now_utc=datetime(2026, 1, 6, 21, 10, tzinfo=UTC),
        baseline_expectancy_bps=2.2,
    )

    promoted_path = result.get("promoted_model_path")
    assert isinstance(promoted_path, str) and promoted_path
    promoted = Path(promoted_path)
    assert promoted.exists()
    assert promoted.read_bytes() == b"rl-model"
    governance_path = result.get("promoted_governance_path")
    assert isinstance(governance_path, str) and governance_path
    governance_payload = json.loads(Path(governance_path).read_text(encoding="utf-8"))
    assert governance_payload["governance_status"] == "production"
    assert governance_payload["recommend_use_rl_agent"] is True


def test_maybe_train_rl_overlay_promotion_permission_denied_is_fail_soft(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import ai_trading.rl_trading.train as train_mod

    n_rows = 140
    dataset = pd.DataFrame(
        {
            "rsi": np.linspace(45.0, 65.0, n_rows),
            "macd": np.linspace(-0.2, 0.3, n_rows),
            "atr": np.linspace(1.0, 2.5, n_rows),
            "vwap": np.linspace(99.0, 105.0, n_rows),
            "sma_50": np.linspace(98.0, 104.0, n_rows),
            "sma_200": np.linspace(95.0, 101.0, n_rows),
            "close": np.linspace(100.0, 110.0, n_rows),
        }
    )

    class DummyTrainer:
        def __init__(
            self,
            *,
            algorithm: str,
            total_timesteps: int,
            eval_freq: int,
            early_stopping_patience: int,
            seed: int,
        ) -> None:
            self.algorithm = algorithm

        def train(
            self,
            *,
            data: np.ndarray,
            env_params: dict[str, object],
            save_path: str,
        ) -> dict[str, object]:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            artifact = save_dir / f"model_{self.algorithm.lower()}.zip"
            artifact.write_bytes(b"rl-model")
            return {
                "final_evaluation": {"mean_reward": 1.0},
                "model_id": "rl-model-perm-denied",
                "governance_status": "production",
            }

    def _raise_permission_error(*_args, **_kwargs):
        raise PermissionError("permission denied during promote")

    monkeypatch.setattr(train_mod, "RLTrainer", DummyTrainer)
    monkeypatch.setattr(after_hours.shutil, "copy2", _raise_permission_error)
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RL_OVERLAY_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RL_TIMESTEPS", "2500")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RL_ALGO", "PPO")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RL_DIR", str(tmp_path / "rl"))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTE_RL_PATH", "1")
    monkeypatch.setenv("AI_TRADING_RL_MODEL_PATH", str(tmp_path / "runtime_rl.zip"))
    monkeypatch.setenv("AI_TRADING_RL_MIN_MEAN_REWARD", "0.0")
    monkeypatch.setenv("AI_TRADING_RL_REQUIRE_BASELINE_EXPECTANCY_BPS", "0.0")
    caplog.set_level("ERROR")

    result = after_hours._maybe_train_rl_overlay(
        dataset,
        now_utc=datetime(2026, 1, 6, 21, 10, tzinfo=UTC),
        baseline_expectancy_bps=2.2,
    )

    assert result["enabled"] is True
    assert result["trained"] is True
    assert result.get("promoted_model_path") is None
    assert any(
        "AFTER_HOURS_RL_PROMOTION_FAILED" in record.message
        for record in caplog.records
    )


def test_maybe_train_rl_overlay_skips_promotion_when_governance_shadow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ai_trading.rl_trading.train as train_mod

    n_rows = 140
    dataset = pd.DataFrame(
        {
            "rsi": np.linspace(45.0, 65.0, n_rows),
            "macd": np.linspace(-0.2, 0.3, n_rows),
            "atr": np.linspace(1.0, 2.5, n_rows),
            "vwap": np.linspace(99.0, 105.0, n_rows),
            "sma_50": np.linspace(98.0, 104.0, n_rows),
            "sma_200": np.linspace(95.0, 101.0, n_rows),
            "close": np.linspace(100.0, 110.0, n_rows),
        }
    )

    class DummyTrainer:
        def __init__(
            self,
            *,
            algorithm: str,
            total_timesteps: int,
            eval_freq: int,
            early_stopping_patience: int,
            seed: int,
        ) -> None:
            self.algorithm = algorithm

        def train(
            self,
            *,
            data: np.ndarray,
            env_params: dict[str, object],
            save_path: str,
        ) -> dict[str, object]:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            artifact = save_dir / f"model_{self.algorithm.lower()}.zip"
            artifact.write_bytes(b"rl-model")
            return {
                "final_evaluation": {"mean_reward": 1.0},
                "model_id": "rl-model-shadow",
                "governance_status": "shadow",
            }

    monkeypatch.setattr(train_mod, "RLTrainer", DummyTrainer)
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RL_OVERLAY_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RL_TIMESTEPS", "2500")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RL_ALGO", "PPO")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RL_DIR", str(tmp_path / "rl"))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTE_RL_PATH", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RL_PROMOTION_REQUIRE_PRODUCTION", "1")
    monkeypatch.setenv("AI_TRADING_RL_MODEL_PATH", str(tmp_path / "runtime_rl.zip"))
    monkeypatch.setenv("AI_TRADING_RL_MIN_MEAN_REWARD", "0.0")
    monkeypatch.setenv("AI_TRADING_RL_REQUIRE_BASELINE_EXPECTANCY_BPS", "0.0")

    result = after_hours._maybe_train_rl_overlay(
        dataset,
        now_utc=datetime(2026, 1, 6, 21, 10, tzinfo=UTC),
        baseline_expectancy_bps=2.2,
        governance_status="shadow",
    )

    assert result["enabled"] is True
    assert result["trained"] is True
    assert result.get("promoted_model_path") is None
    assert result.get("promoted_governance_path") is None


def test_on_market_close_applies_promoted_model_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ai_trading.core import bot_engine

    class _FixedDateTime:
        @staticmethod
        def now(_tz=None):
            return datetime(2026, 1, 6, 21, 10, tzinfo=UTC)

    calls: dict[str, list[dict[str, object]]] = {"ml": [], "rl": []}
    marker_path = tmp_path / "after_hours.marker.json"
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_ONCE_PER_DAY", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_MARKER_PATH", str(marker_path))
    monkeypatch.setenv("AI_TRADING_LEGACY_DAILY_RETRAIN_ENABLED", "0")
    monkeypatch.setattr(bot_engine, "dt_", _FixedDateTime)
    monkeypatch.setattr(bot_engine, "market_is_open", lambda *_args, **_kwargs: False)
    def _refresh_required_model_cache(path, manifest_path=None, reason="runtime"):
        calls["ml"].append({"path": path, "manifest_path": manifest_path, "reason": reason})
        return True

    monkeypatch.setattr(
        bot_engine,
        "_refresh_required_model_cache_from_path",
        _refresh_required_model_cache,
    )
    def _reload_rl_agent(model_path=None, reason="runtime", force=False, use_rl_enabled=None):
        calls["rl"].append(
            {
                "model_path": model_path,
                "reason": reason,
                "force": force,
                "use_rl_enabled": use_rl_enabled,
            }
        )
        return True

    monkeypatch.setattr(
        bot_engine,
        "_reload_rl_agent_from_runtime_path",
        _reload_rl_agent,
    )
    monkeypatch.setattr(
        after_hours,
        "run_after_hours_training",
        lambda **_kwargs: {
            "status": "trained",
            "model_id": "m-1",
            "model_name": "logreg",
            "promoted_model_path": "/tmp/ml-promoted.joblib",
            "promoted_manifest_path": "/tmp/ml-promoted.manifest.json",
            "rl_overlay": {"promoted_model_path": "/tmp/rl-promoted.zip"},
        },
    )

    bot_engine.on_market_close()

    assert calls["ml"] == [
        {
            "path": "/tmp/ml-promoted.joblib",
            "manifest_path": "/tmp/ml-promoted.manifest.json",
            "reason": "after_hours_training",
        }
    ]
    assert calls["rl"] == [
        {
            "model_path": "/tmp/rl-promoted.zip",
            "reason": "after_hours_training",
            "force": True,
            "use_rl_enabled": None,
        }
    ]
