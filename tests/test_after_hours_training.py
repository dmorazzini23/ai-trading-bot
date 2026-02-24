from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai_trading.training import after_hours


pytest.importorskip("sklearn")


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
    rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
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

    monkeypatch.setenv("AI_TRADING_TICKERS_CSV", str(tickers))
    monkeypatch.setenv("AI_TRADING_TCA_PATH", str(tca_path))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_REPORT_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("AI_TRADING_MODEL_PATH", str(tmp_path / "runtime_model.joblib"))
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTE_MODEL_PATH", "1")
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

    result = after_hours.run_after_hours_training(
        now=datetime(2026, 1, 6, 21, 10, tzinfo=UTC),  # 16:10 New York
    )
    assert result["status"] == "trained"
    assert Path(result["model_path"]).exists()
    assert Path(result["manifest_path"]).exists()
    assert Path(result["report_path"]).exists()
    assert Path(result["promoted_model_path"]).exists()
    assert Path(result["promoted_manifest_path"]).exists()

    report_payload = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
    assert "metrics" in report_payload
    assert "thresholds_by_regime" in report_payload
    assert "candidate_metrics" in report_payload
    assert "sensitivity_sweep" in report_payload
    assert "manifest_metadata" in report_payload["model"]
    assert isinstance(report_payload["candidate_metrics"], list)
    assert report_payload["candidate_metrics"]
    assert "candidate_metrics" in result
    assert isinstance(result["candidate_metrics"], list)
    assert result["candidate_metrics"]
    assert "sensitivity_sweep" in result

    manifest_payload = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert "metadata" in manifest_payload
    assert manifest_payload["metadata"]["strategy"] == "after_hours_ml_edge"


def test_after_hours_sensitivity_gate_can_block_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tickers = tmp_path / "tickers.csv"
    tickers.write_text("symbol\nAAPL\nMSFT\n", encoding="utf-8")
    tca_path = tmp_path / "tca_records.jsonl"
    _write_tca(tca_path, n=420)

    monkeypatch.setenv("AI_TRADING_TICKERS_CSV", str(tickers))
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
    monkeypatch.setattr(
        after_hours,
        "run_after_hours_training",
        lambda **_kwargs: calls.__setitem__("after_hours", calls["after_hours"] + 1)
        or {"status": "trained", "model_id": "m-1", "model_name": "logreg"},
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
    monkeypatch.setattr(
        after_hours,
        "run_after_hours_training",
        lambda **_kwargs: calls.__setitem__("after_hours", calls["after_hours"] + 1)
        or {"status": "trained"},
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

    monkeypatch.setenv("AI_TRADING_TICKERS_CSV", str(tickers))
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

    monkeypatch.setenv("AI_TRADING_TICKERS_CSV", str(tickers))
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

    monkeypatch.setenv("AI_TRADING_TICKERS_CSV", str(live_tickers))
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
    monkeypatch.setattr(
        bot_engine,
        "_refresh_required_model_cache_from_path",
        lambda path, manifest_path=None, reason="runtime": calls["ml"].append(
            {"path": path, "manifest_path": manifest_path, "reason": reason}
        )
        or True,
    )
    monkeypatch.setattr(
        bot_engine,
        "_reload_rl_agent_from_runtime_path",
        lambda model_path=None, reason="runtime", force=False, use_rl_enabled=None: calls[
            "rl"
        ].append(
            {
                "model_path": model_path,
                "reason": reason,
                "force": force,
                "use_rl_enabled": use_rl_enabled,
            }
        )
        or True,
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
