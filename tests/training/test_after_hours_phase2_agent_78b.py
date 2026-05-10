from __future__ import annotations

import json
import sys
import types
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ai_trading.training import after_hours as ah


def _install_env(monkeypatch: pytest.MonkeyPatch, values: dict[str, Any]) -> None:
    def fake_get_env(
        name: str,
        default: Any = None,
        *,
        cast: Any = None,
        resolve_aliases: bool = True,
    ) -> Any:
        del resolve_aliases
        raw = values.get(name, default)
        if cast is None or raw is None:
            return raw
        if cast is bool:
            if isinstance(raw, str):
                return raw.strip().lower() not in {"0", "false", "no", "off", ""}
            return bool(raw)
        return cast(raw)

    monkeypatch.setattr(ah, "get_env", fake_get_env)


def _candidate(
    name: str,
    *,
    expectancy: float,
    support: int = 80,
    profitable_folds: int = 3,
    profitable_ratio: float = 0.75,
) -> ah.CandidateMetrics:
    return ah.CandidateMetrics(
        name=name,
        fold_count=4,
        profitable_fold_count=profitable_folds,
        profitable_fold_ratio=profitable_ratio,
        support=support,
        mean_expectancy_bps=expectancy,
        max_drawdown_bps=20.0,
        turnover_ratio=0.12,
        mean_hit_rate=0.62,
        hit_rate_stability=0.85,
        regime_metrics={},
        oof_probabilities=np.array([0.2, 0.8]),
        fold_expectancy_bps=(expectancy - 1.0, expectancy, expectancy + 1.0),
        brier_score=0.18,
    )


def _tiny_dataset(rows: int = 3) -> pd.DataFrame:
    base = pd.Timestamp("2026-04-20T00:00:00Z")
    return pd.DataFrame(
        {
            "timestamp": [base + pd.Timedelta(days=idx) for idx in range(rows)],
            "label_ts": [base + pd.Timedelta(days=idx + 1) for idx in range(rows)],
            "label": [0, 1, 0][:rows],
            "realized_edge_bps": [-2.0, 5.0, -1.0][:rows],
            "regime": ["sideways", "uptrend", "sideways"][:rows],
            "symbol": ["AAPL", "MSFT", "AAPL"][:rows],
        }
    )


def test_jsonl_tca_parsing_and_cost_floor_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = datetime.now(UTC)
    records_path = tmp_path / "tca.jsonl"
    rows = [
        "",
        "{not json",
        json.dumps(["not", "a", "dict"]),
        json.dumps({"is_bps": "bad", "status": "filled"}),
        json.dumps({"is_bps": 6, "status": "filled", "ts": now.isoformat()}),
        json.dumps({"is_bps": 10, "status": "partially_filled", "fill_latency_ms": "7"}),
        json.dumps({"is_bps": 14, "status": "canceled"}),
        json.dumps({"is_bps": 200, "status": "filled"}),
    ]
    records_path.write_text("\n".join(rows), encoding="utf-8")

    assert ah._read_jsonl_records(str(tmp_path / "missing.jsonl")) == []
    assert [row.get("is_bps") for row in ah._read_jsonl_records(str(records_path), max_records=2)] == [
        14,
        200,
    ]
    assert ah._parse_ts("") is None
    assert ah._parse_ts("not-a-date") is None
    assert ah._parse_ts("2026-04-20T12:00:00Z") == datetime(
        2026, 4, 20, 12, 0, tzinfo=UTC
    )
    assert ah._parse_ts("2026-04-20T12:00:00").tzinfo == UTC
    assert ah._tca_row_timestamp({"fill_ts": "2026-04-20T12:00:00Z"}) is not None
    assert ah._tca_row_timestamp({"other": "2026-04-20T12:00:00Z"}) is None

    fill_quality = ah._build_fill_quality_metrics(ah._read_jsonl_records(str(records_path)))
    assert fill_quality["fill_rate"] == pytest.approx(3 / 5)
    assert fill_quality["partial_fill_rate"] == 0.0
    assert fill_quality["mean_fill_latency_ms"] == 7.0

    _install_env(
        monkeypatch,
        {
            "AI_TRADING_COST_FLOOR_BPS": 12,
            "AI_TRADING_AFTER_HOURS_COST_FLOOR_MIN_BPS": 4,
            "AI_TRADING_AFTER_HOURS_COST_FLOOR_MAX_BPS": 25,
            "AI_TRADING_AFTER_HOURS_COST_FLOOR_LOOKBACK_DAYS": 3650,
            "AI_TRADING_AFTER_HOURS_COST_FLOOR_MIN_SAMPLES": 2,
            "AI_TRADING_AFTER_HOURS_COST_FLOOR_REQUIRE_FILLED": True,
            "AI_TRADING_AFTER_HOURS_COST_FLOOR_OUTLIER_BPS": 100,
            "AI_TRADING_AFTER_HOURS_COST_FLOOR_QUANTILE": 0.5,
            "AI_TRADING_AFTER_HOURS_COST_FLOOR_TCA_WEIGHT": 1.0,
        },
    )
    assert ah._estimate_cost_floor_bps(ah._read_jsonl_records(str(records_path))) == 8.0
    assert ah._estimate_cost_floor_bps([]) == 12.0


def test_output_and_model_write_fallbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested = tmp_path / "requested"
    fallback = tmp_path / "fallback"
    access_calls: list[Path] = []

    def fake_access(path: Path, mode: int) -> bool:
        del mode
        access_calls.append(Path(path))
        return Path(path) == fallback

    monkeypatch.setattr(ah.os, "access", fake_access)
    assert (
        ah._resolve_writable_output_dir(
            requested=requested,
            fallback=fallback,
            event_name="TEST_FALLBACK",
        )
        == fallback
    )
    assert access_calls == [requested, fallback]

    monkeypatch.setattr(ah.os, "access", lambda path, mode: False)
    with pytest.raises(RuntimeError, match="No writable directory"):
        ah._resolve_writable_output_dir(
            requested=tmp_path / "nope",
            fallback=tmp_path / "still_nope",
            event_name="TEST_UNWRITABLE",
        )

    import joblib

    dumps: list[Path] = []

    def fake_dump(model: object, path: Path) -> None:
        del model
        dumps.append(Path(path))
        if len(dumps) == 1:
            raise OSError(30, "read-only")

    monkeypatch.setattr(joblib, "dump", fake_dump)
    monkeypatch.setattr(ah.os, "access", lambda path, mode: True)
    model_path = tmp_path / "primary" / "model.joblib"
    fallback_dir = tmp_path / "models"
    assert ah._dump_model_with_fallback(object(), model_path, fallback_dir=fallback_dir) == (
        fallback_dir / "model.joblib"
    )
    assert dumps == [model_path, fallback_dir / "model.joblib"]


def test_probability_fallback_model_and_orientation(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame({"alpha": [0.0, 1.0, 2.0, 3.0], "beta": [1.0, 1.0, 0.0, 0.0]})
    model = ah._FallbackProbabilityModel().fit(frame, [0, 0, 1, 1])
    probs = model.predict_proba(frame)
    assert probs.shape == (4, 2)
    assert model.predict(frame).shape == (4,)
    assert model.feature_names_in_.tolist() == ["alpha", "beta"]
    with pytest.raises(RuntimeError, match="Model not fitted"):
        ah._FallbackProbabilityModel().predict_proba(frame)

    class ReversedClassModel:
        classes_ = np.array([1, 0])

        def predict_proba(self, X: object) -> np.ndarray:
            return np.array([[0.9, 0.1], [0.2, 0.8]])

    assert ah._predict_probabilities(ReversedClassModel(), frame).tolist() == [0.9, 0.2]

    class DecisionOnly:
        def decision_function(self, X: object) -> np.ndarray:
            return np.array([-2.0, 2.0])

    assert ah._predict_probabilities(DecisionOnly(), frame).tolist() == pytest.approx(
        [0.1192029, 0.8807971]
    )

    class PredictOnly:
        def predict(self, X: object) -> np.ndarray:
            return np.array([-1.0, 0.5, 2.0])

    assert ah._predict_probabilities(PredictOnly(), frame).tolist() == [0.0, 0.5, 1.0]

    _install_env(
        monkeypatch,
        {
            "AI_TRADING_AFTER_HOURS_SCORE_ORIENTATION_QUANTILE": 0.5,
            "AI_TRADING_AFTER_HOURS_SCORE_ORIENTATION_MIN_DELTA_BPS": 0.1,
        },
    )
    oriented, report = ah._orient_probabilities_for_edge(
        np.array([0.1, 0.2, 0.8, 0.9]),
        np.array([8.0, 6.0, -2.0, -4.0]),
    )
    assert report["orientation"] == "inverse"
    assert report["inverse_applied"] is True
    assert oriented.tolist() == pytest.approx([0.9, 0.8, 0.2, 0.1])


def test_report_snapshot_and_prior_metrics_parsing(tmp_path: Path) -> None:
    assert ah._report_candidate_from_payload({"name": ""}) is None
    assert ah._report_snapshot_from_payload({"ts": "bad"}, source="bad") is None

    payload = {
        "ts": "2026-04-20T23:00:00Z",
        "model": {},
        "metrics": {"mean_expectancy_bps": "bad", "hit_rate_stability": 0.7},
        "candidate_metrics": [
            {"not": "mapping"},
            {
                "name": "logreg",
                "selected": True,
                "mean_expectancy_bps": 3.0,
                "max_drawdown_bps": 11.0,
                "turnover_ratio": 0.1,
                "hit_rate_stability": 0.55,
                "brier_score": 0.2,
                "support": -4,
                "profitable_fold_ratio": 0.5,
            },
        ],
    }
    snapshot = ah._report_snapshot_from_payload(payload, source="pending")
    assert snapshot is not None
    assert snapshot.model_name == "logreg"
    assert snapshot.mean_expectancy_bps == 3.0
    assert snapshot.candidates[0].support == 0

    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    valid_report = report_dir / "after_hours_training_20260420_230000.json"
    valid_report.write_text(json.dumps(payload), encoding="utf-8")
    (report_dir / "after_hours_training_20260421_230000.json").write_text("{bad", encoding="utf-8")
    duplicate_pending = {**payload, "metrics": {"mean_expectancy_bps": 99.0}}
    snapshots = ah._recent_report_snapshots(
        report_dir=report_dir,
        max_reports=4,
        pending_report=duplicate_pending,
    )
    assert len(snapshots) == 1
    assert snapshots[0].source == str(valid_report)

    prior_payload = {
        "ts": "2026-04-20T23:00:00Z",
        "model": {"model_id": "m1", "governance_status": "shadow"},
        "metrics": {
            "mean_expectancy_bps": 4,
            "max_drawdown_bps": 20,
            "mean_hit_rate": 0.6,
            "fold_count": 3,
        },
        "candidate_metrics": [
            {"name": "ranked", "rank": 1, "fold_expectancy_bps": ["1.5", "bad", 2.5]},
            {"name": "other", "fold_expectancy_bps": [9]},
        ],
        "runtime_performance_gate": {
            "thresholds": {"min_closed_trades": 10},
            "observed": {"execution_capture_ratio": 0.3, "slippage_drag_bps": 8.0},
        },
        "live_execution_quality_gate": {
            "observed": {"execution_capture_ratio": 0.4, "slippage_drag_bps": 6.0}
        },
    }
    parsed = ah._prior_metrics_from_report_payload(prior_payload, report_path=valid_report)
    assert parsed is not None
    assert parsed["fold_expectancy_bps"] == (1.5, 2.5)
    assert parsed["runtime_execution_quality"] == {
        "execution_capture_ratio": 0.4,
        "slippage_drag_bps": 6.0,
    }
    assert ah._report_is_production({"promotion": {"status": "production"}}) is True


def test_prior_metrics_loader_prefers_distinct_production_report(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    older_production = {
        "ts": "2026-04-20T23:00:00Z",
        "model": {"model_id": "prod", "governance_status": "production"},
        "metrics": {"mean_expectancy_bps": 5, "max_drawdown_bps": 10},
    }
    latest_shadow = {
        "ts": "2026-04-21T23:00:00Z",
        "model": {"model_id": "latest", "governance_status": "shadow"},
        "metrics": {"mean_expectancy_bps": 1, "max_drawdown_bps": 40},
    }
    (report_dir / "after_hours_training_20260420_230000.json").write_text(
        json.dumps(older_production),
        encoding="utf-8",
    )
    (report_dir / "after_hours_training_20260421_230000.json").write_text(
        json.dumps(latest_shadow),
        encoding="utf-8",
    )

    prior = ah._load_prior_model_metrics(report_dir=report_dir)
    assert prior is not None
    assert prior["model_id"] == "prod"
    assert "is_production" not in prior


def test_selection_constraints_and_weight_retune_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_env(
        monkeypatch,
        {
            "AI_TRADING_AFTER_HOURS_MODEL_SELECTION_CONSTRAINTS_ENABLED": True,
            "AI_TRADING_AFTER_HOURS_MODEL_SELECTION_MIN_PROFITABLE_FOLDS": 2,
            "AI_TRADING_AFTER_HOURS_MODEL_SELECTION_MIN_PROFITABLE_FOLD_RATIO": 0.6,
            "AI_TRADING_AFTER_HOURS_MODEL_SELECTION_MIN_EXPECTANCY_BPS": 0.0,
            "AI_TRADING_AFTER_HOURS_MODEL_SELECTION_PROFITABLE_FOLD_TARGET": 0.6,
            "AI_TRADING_AFTER_HOURS_RETUNE_BASELINE_WINDOW": 2,
            "AI_TRADING_AFTER_HOURS_RETUNE_RECENT_WINDOW": 2,
            "AI_TRADING_AFTER_HOURS_RETUNE_MIN_REPORTS": 4,
            "AI_TRADING_AFTER_HOURS_RETUNE_BRIER_DRIFT_PCT": 0.1,
            "AI_TRADING_AFTER_HOURS_RETUNE_BRIER_ABS_THRESHOLD": 0.25,
            "AI_TRADING_AFTER_HOURS_RETUNE_EXPECTANCY_DROP_BPS": 2.0,
            "AI_TRADING_AFTER_HOURS_RETUNE_STABILITY_DROP": 0.05,
            "AI_TRADING_AFTER_HOURS_RETUNE_SEARCH_ROUNDS": 1,
            "AI_TRADING_AFTER_HOURS_RETUNE_SEARCH_FACTORS": "0.5,1.0,2.0",
            "AI_TRADING_AFTER_HOURS_RETUNE_MIN_UTILITY_IMPROVEMENT": 0.0,
            "AI_TRADING_AFTER_HOURS_RETUNE_MIN_WEIGHT_CHANGE_PCT": 0.0,
            "AI_TRADING_AFTER_HOURS_RETUNE_WEIGHT_REGULARIZATION": 0.0,
        },
    )
    eligible, diagnostics = ah._filter_candidates_for_selection(
        [
            _candidate("good", expectancy=4.0),
            _candidate("bad", expectancy=-5.0, profitable_folds=0, profitable_ratio=0.1),
        ]
    )
    assert [item.name for item in eligible] == ["good"]
    assert diagnostics["rejected_by_reason"] == {
        "expectancy": 1,
        "profitable_fold_ratio": 1,
        "profitable_folds": 1,
    }

    fallback, fallback_diag = ah._filter_candidates_for_selection(
        [
            _candidate("bad1", expectancy=-5.0, profitable_folds=0, profitable_ratio=0.1),
            _candidate("bad2", expectancy=-4.0, profitable_folds=0, profitable_ratio=0.2),
        ]
    )
    assert [item.name for item in fallback] == ["bad1", "bad2"]
    assert fallback_diag["fallback_to_unfiltered"] is True

    reports = [
        ah._ReportSnapshot(
            ts=datetime(2026, 4, 20 + idx, tzinfo=UTC),
            source=str(idx),
            model_name="candidate",
            mean_expectancy_bps=10.0 if idx < 2 else 4.0,
            hit_rate_stability=0.8 if idx < 2 else 0.65,
            brier_score=0.18 if idx < 2 else 0.35,
            candidates=[
                ah._CandidateSnapshot(
                    name="candidate",
                    mean_expectancy_bps=10.0 - idx,
                    max_drawdown_bps=10.0,
                    turnover_ratio=0.1,
                    hit_rate_stability=0.7,
                    brier_score=0.2,
                    support=100,
                    profitable_fold_ratio=0.8,
                )
            ],
        )
        for idx in range(4)
    ]
    drift = ah._drift_breach_summary(reports)
    assert drift["ready"] is True
    assert drift["breached"] is True
    retuned = ah._retune_selection_weights(
        reports,
        current_weights=ah._model_selection_default_weights(),
    )
    assert set(retuned["weights"]) == set(ah._model_selection_weight_bounds())
    safety = ah._apply_selection_weight_safety_envelope(
        current_weights={"turnover_penalty": 1.0},
        candidate_weights={"turnover_penalty": 9.0},
    )
    assert safety["clamped"] is True


def test_promotion_state_and_new_rows_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_env(
        monkeypatch,
        {"AI_TRADING_AFTER_HOURS_PROMOTION_MIN_CONSECUTIVE_PASSES": 2},
    )
    promotion = {
        "gate_passed": True,
        "auto_promote": True,
        "combined_gates": {"edge": True},
        "additional_gates": {},
    }
    updated, state = ah._apply_promotion_consecutive_pass_gate(
        promotion=promotion,
        previous_state={
            "promotion_consecutive_passes": "1",
            "promotion_last_pass_date": "2026-04-20",
        },
        run_date=date(2026, 4, 21),
    )
    assert updated["status"] == "production"
    assert updated["combined_gates"]["consecutive_passes"] is True
    assert state["count"] == 2

    reset, reset_state = ah._apply_promotion_consecutive_pass_gate(
        promotion={**promotion, "gate_passed": False},
        previous_state={"promotion_consecutive_passes": "bad", "updated_at": "2026-04-20T22:00:00Z"},
        run_date=date(2026, 4, 21),
    )
    assert reset["consecutive_passes"]["reason"] == "reset_gate_failed"
    assert reset_state["count"] == 0
    assert ah._parse_iso_date("2026-04-21T23:59:00Z") == date(2026, 4, 21)
    assert ah._parse_iso_date("bad") is None

    inverse_candidate = _candidate("inverse", expectancy=5.0)
    inverse_candidate.score_orientation = "inverse"
    inverse_candidate.score_orientation_report = {"inverse_applied": True}
    inverse_guard = ah._oof_promotion_authority_guard(
        best=inverse_candidate,
        default_threshold=0.5,
        live_execution_quality_gate={},
        champion_challenger_ab={},
        promotion_confidence_gate={},
    )
    assert inverse_guard["gate_passed"] is False
    assert inverse_guard["reasons"] == ["inverse_global_oof_orientation_shadow_only"]

    tuned_candidate = _candidate("tuned", expectancy=5.0)
    tuned_candidate.selected_threshold = 0.65
    tuned_guard = ah._oof_promotion_authority_guard(
        best=tuned_candidate,
        default_threshold=0.5,
        live_execution_quality_gate={},
        champion_challenger_ab={},
        promotion_confidence_gate={},
    )
    assert tuned_guard["gate_passed"] is False
    assert "threshold_tuning_requires_holdout_or_live_shadow_confirmation" in tuned_guard["reasons"]
    confirmed_guard = ah._oof_promotion_authority_guard(
        best=tuned_candidate,
        default_threshold=0.5,
        live_execution_quality_gate={"gate_passed": True},
        champion_challenger_ab={},
        promotion_confidence_gate={},
    )
    assert confirmed_guard["gate_passed"] is True

    sanitized = ah._sanitize_after_hours_training_state_payload(
        {
            "report_path": "/tmp/pytest-of-aiuser/run/report.json",
            "daily_report_path": "/var/lib/ai-trading/report.json",
        }
    )
    assert "report_path" not in sanitized
    assert "daily_report_path" in sanitized

    dataset = _tiny_dataset()
    assert ah._new_rows_since_training_state(dataset, previous_state=None) == 3
    assert (
        ah._new_rows_since_training_state(
            dataset,
            previous_state={"max_label_ts": "2026-04-21T00:00:00Z"},
        )
        == 2
    )
    assert ah._new_rows_since_training_state(dataset.iloc[0:0], previous_state=None) == 0


def test_training_state_load_write_fallbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preferred = tmp_path / "preferred.json"
    default = tmp_path / "default.json"
    preferred.write_text("{bad", encoding="utf-8")
    default.write_text(
        json.dumps({"report_path": "/tmp/pytest-run/report.json", "rows": 10}),
        encoding="utf-8",
    )
    path_calls = iter([default, preferred, default])
    monkeypatch.setattr(ah, "_after_hours_training_state_path", lambda **kwargs: next(path_calls))
    loaded = ah._load_after_hours_training_state(preferred_path=preferred)
    assert loaded == {"rows": 10}

    monkeypatch.setattr(ah, "_after_hours_training_state_path", lambda **kwargs: preferred)
    monkeypatch.setattr(
        ah,
        "_resolve_writable_output_dir",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("no writable")),
    )
    assert ah._write_after_hours_training_state({"rows": 1}, preferred_path=preferred) is None


def test_runtime_performance_gate_fail_open_and_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_env(
        monkeypatch,
        {
            "AI_TRADING_AFTER_HOURS_PROMOTION_RUNTIME_GONOGO_ENABLED": True,
            "AI_TRADING_TRADE_HISTORY_PATH": str(tmp_path / "trades.parquet"),
            "AI_TRADING_RUNTIME_PERF_GATE_SUMMARY_PATH": str(tmp_path / "gate.json"),
            "AI_TRADING_AFTER_HOURS_PROMOTION_RUNTIME_GONOGO_FAIL_ON_DATA_UNAVAILABLE": False,
            "AI_TRADING_AFTER_HOURS_PROMOTION_RUNTIME_GONOGO_MIN_REALIZED_TRADES_FOR_DATA": 5,
        },
    )

    fake_report_module = types.SimpleNamespace()

    def build_report(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {"execution_vs_alpha": {"execution_capture_ratio": 0.2}}

    def evaluate_go_no_go(report: dict[str, Any], *, thresholds: dict[str, Any]) -> dict[str, Any]:
        del report
        return {
            "gate_passed": False,
            "reason": "failed",
            "checks": {"closed_trades": False, "pnl_available": False},
            "failed_checks": ["closed_trades", "pnl_available"],
            "observed": {
                "closed_trades": 0,
                "trade_used_days": 0,
                "pnl_available": False,
            },
            "thresholds": thresholds,
        }

    fake_report_module.build_report = build_report
    fake_report_module.evaluate_go_no_go = evaluate_go_no_go
    import ai_trading.tools as tools_pkg

    monkeypatch.setattr(tools_pkg, "runtime_performance_report", fake_report_module, raising=False)
    result = ah._runtime_performance_go_no_go_gate()
    assert result["gate_passed"] is True
    assert result["reason"] == "insufficient_runtime_data_fail_open"
    assert result["soft_failed_checks"] == ["closed_trades", "pnl_available"]

    def raising_build_report(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        raise RuntimeError("boom")

    fake_report_module.build_report = raising_build_report
    failed = ah._runtime_performance_go_no_go_gate()
    assert failed["gate_passed"] is False
    assert failed["reason"] == "runtime_performance_eval_failed"


def test_rl_overlay_training_with_stubbed_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_env(
        monkeypatch,
        {
            "AI_TRADING_AFTER_HOURS_RL_OVERLAY_ENABLED": True,
            "AI_TRADING_AFTER_HOURS_RL_TIMESTEPS": 2500,
            "AI_TRADING_AFTER_HOURS_RL_ALGO": "ppo",
            "AI_TRADING_AFTER_HOURS_RL_DIR": str(tmp_path / "rl"),
            "AI_TRADING_SEED": 7,
            "AI_TRADING_RL_MIN_MEAN_REWARD": 1.0,
            "AI_TRADING_RL_REQUIRE_BASELINE_EXPECTANCY_BPS": 2.0,
            "AI_TRADING_AFTER_HOURS_RL_MULTI_SEED_ENABLED": True,
            "AI_TRADING_AFTER_HOURS_RL_MULTI_SEEDS": "11,bad,23,11",
            "AI_TRADING_AFTER_HOURS_RL_PROMOTION_REQUIRE_PRODUCTION": True,
            "AI_TRADING_AFTER_HOURS_RL_PROMOTION_REQUIRE_RECOMMEND": True,
        },
    )

    class FakeTrainer:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def train(self, *, data: np.ndarray, env_params: dict[str, Any], save_path: str) -> dict[str, Any]:
            assert data.shape[0] == 130
            assert env_params["dataset_fingerprint"] == "fp"
            run_dir = Path(save_path)
            run_dir.mkdir(parents=True)
            (run_dir / "model_ppo.zip").write_bytes(b"model")
            return {
                "final_evaluation": {"mean_reward": 3.5},
                "model_id": "rl-1",
                "governance_status": "production",
            }

    def fake_train_multi_seed(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["seeds"] == [11, 23]
        return {"runs": 2}

    train_module = types.ModuleType("ai_trading.rl_trading.train")
    train_module.RLTrainer = FakeTrainer
    train_module.train_multi_seed = fake_train_multi_seed
    monkeypatch.setitem(sys.modules, "ai_trading.rl_trading.train", train_module)
    monkeypatch.setattr(
        ah,
        "_maybe_promote_rl_overlay_to_runtime_path",
        lambda trained_model_path: str(tmp_path / "runtime" / "rl_agent.zip"),
    )
    monkeypatch.setattr(
        ah,
        "_write_rl_runtime_governance_sidecar",
        lambda runtime_model_path, **kwargs: str(Path(f"{runtime_model_path}.governance.json")),
    )

    dataset = pd.DataFrame(
        {
            **{
                feature: np.linspace(0.0, 1.0, 130)
                for feature in ah.FEATURE_COLUMNS
            },
            "close": np.linspace(100.0, 110.0, 130),
        }
    )
    result = ah._maybe_train_rl_overlay(
        dataset,
        now_utc=datetime(2026, 4, 21, 23, 0, tzinfo=UTC),
        baseline_expectancy_bps=2.5,
        dataset_fingerprint="fp",
        feature_hash="fh",
        governance_status="production",
    )
    assert result["trained"] is True
    assert result["recommend_use_rl_agent"] is True
    assert result["promoted_model_path"] == str(tmp_path / "runtime" / "rl_agent.zip")
    assert result["promoted_governance_path"] == str(
        tmp_path / "runtime" / "rl_agent.zip.governance.json"
    )
    assert result["multi_seed_summary"] == {"runs": 2}


def test_run_after_hours_training_scheduling_skips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_env(
        monkeypatch,
        {
            "AI_TRADING_AFTER_HOURS_TRAINING_CATCHUP_ENABLED": True,
            "AI_TRADING_AFTER_HOURS_MIN_ROWS": 5,
            "AI_TRADING_TCA_PATH": str(tmp_path / "missing-tca.jsonl"),
            "AI_TRADING_AFTER_HOURS_TCA_WINDOW": 10,
            "AI_TRADING_COST_FLOOR_BPS": 12.0,
            "AI_TRADING_AFTER_HOURS_COST_FLOOR_MIN_BPS": 4.0,
            "AI_TRADING_AFTER_HOURS_COST_FLOOR_MAX_BPS": 25.0,
            "AI_TRADING_AFTER_HOURS_LOOKBACK_DAYS": 180,
            "AI_TRADING_AFTER_HOURS_REPORT_DIR": str(tmp_path / "reports"),
            "AI_TRADING_AFTER_HOURS_MIN_NEW_ROWS_FOR_RETRAIN": 2,
            "AI_TRADING_AFTER_HOURS_SKIP_IF_NO_NEW_SIGNAL_DATA": True,
            "AI_TRADING_AFTER_HOURS_FORCE_RETRAIN": False,
        },
    )
    monkeypatch.setattr(ah, "reload_env", lambda override=False: None)
    monkeypatch.setattr(ah, "refresh_default_feed", lambda: None)

    before_close = ah.run_after_hours_training(now=datetime(2026, 4, 21, 16, 0, tzinfo=UTC))
    assert before_close["reason"] == "before_market_close"

    monkeypatch.setattr(ah, "_load_symbols", lambda: [])
    no_symbols = ah.run_after_hours_training(now=datetime(2026, 4, 21, 22, 0, tzinfo=UTC))
    assert no_symbols["reason"] == "no_symbols"

    monkeypatch.setattr(ah, "_load_symbols", lambda: ["AAPL"])
    monkeypatch.setattr(ah, "_build_training_dataset", lambda *args, **kwargs: pd.DataFrame())
    insufficient = ah.run_after_hours_training(now=datetime(2026, 4, 21, 22, 0, tzinfo=UTC))
    assert insufficient["reason"] == "insufficient_dataset"
    assert insufficient["required_rows"] == 5

    dataset = _tiny_dataset(rows=3)
    monkeypatch.setattr(ah, "_build_training_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(ah, "_dataset_fingerprint", lambda *args, **kwargs: "same-fp")
    monkeypatch.setattr(
        ah,
        "_load_after_hours_training_state",
        lambda **kwargs: {
            "dataset_fingerprint": "same-fp",
            "max_label_ts": (
                dataset["label_ts"].max() - pd.Timedelta(days=1)
            ).isoformat(),
        },
    )
    _install_env(
        monkeypatch,
        {
            "AI_TRADING_AFTER_HOURS_TRAINING_CATCHUP_ENABLED": True,
            "AI_TRADING_AFTER_HOURS_MIN_ROWS": 3,
            "AI_TRADING_TCA_PATH": str(tmp_path / "missing-tca.jsonl"),
            "AI_TRADING_AFTER_HOURS_TCA_WINDOW": 10,
            "AI_TRADING_COST_FLOOR_BPS": 12.0,
            "AI_TRADING_AFTER_HOURS_COST_FLOOR_MIN_BPS": 4.0,
            "AI_TRADING_AFTER_HOURS_COST_FLOOR_MAX_BPS": 25.0,
            "AI_TRADING_AFTER_HOURS_LOOKBACK_DAYS": 180,
            "AI_TRADING_AFTER_HOURS_REPORT_DIR": str(tmp_path / "reports"),
            "AI_TRADING_AFTER_HOURS_MIN_NEW_ROWS_FOR_RETRAIN": 2,
            "AI_TRADING_AFTER_HOURS_SKIP_IF_NO_NEW_SIGNAL_DATA": True,
            "AI_TRADING_AFTER_HOURS_FORCE_RETRAIN": False,
        },
    )
    no_new = ah.run_after_hours_training(now=datetime(2026, 4, 21, 22, 0, tzinfo=UTC))
    assert no_new["reason"] == "no_new_signal_data"
    assert no_new["unchanged_dataset_fingerprint"] is True
