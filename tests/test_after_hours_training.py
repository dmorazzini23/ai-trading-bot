from __future__ import annotations

import json
from datetime import UTC, datetime
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


def test_on_market_close_runs_after_hours_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ai_trading.core import bot_engine

    calls: dict[str, int] = {"after_hours": 0, "legacy": 0}

    class _FixedDateTime:
        @staticmethod
        def now(_tz=None):
            return datetime(2026, 1, 6, 21, 10, tzinfo=UTC)

    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_ENABLED", "1")
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
