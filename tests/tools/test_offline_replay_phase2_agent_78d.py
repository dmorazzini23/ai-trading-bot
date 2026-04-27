from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ai_trading.tools import offline_replay as replay
from ai_trading.tools import replay_governance


def _cfg(**overrides: Any) -> replay.ReplayConfig:
    values = {
        "confidence_threshold": 0.0,
        "entry_score_threshold": 0.0,
        "allow_shorts": True,
        "min_hold_bars": 1,
        "max_hold_bars": 3,
        "stop_loss_bps": 1_000.0,
        "take_profit_bps": 1_000.0,
        "trailing_stop_bps": 1_000.0,
        "fee_bps": 0.0,
        "slippage_bps": 0.0,
    }
    values.update(overrides)
    return replay.ReplayConfig(**values)


def _args(**overrides: Any) -> argparse.Namespace:
    values = {
        "csv": None,
        "data_dir": None,
        "symbol": "",
        "symbols": "",
        "timestamp_col": "timestamp",
        "confidence_threshold": 0.0,
        "entry_score_threshold": 0.0,
        "allow_shorts": True,
        "min_hold_bars": 1,
        "max_hold_bars": 3,
        "stop_loss_bps": 10.0,
        "take_profit_bps": 10.0,
        "trailing_stop_bps": 10.0,
        "fee_bps": 0.0,
        "slippage_bps": 0.0,
        "simulation_mode": False,
        "replay_seed": 1,
        "max_symbol_notional": None,
        "max_gross_notional": None,
        "persist_intents": False,
        "intent_prefix": "replay",
        "policy_sensitivity_mode": False,
        "apply_policy_controls": False,
        "use_model_score": False,
        "model_path": None,
        "output_json": None,
        "env_file": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _frame(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2026-01-02T14:30:00Z", periods=len(closes), freq="min")
    return pd.DataFrame(
        {
            "open": closes,
            "high": [value + 0.1 for value in closes],
            "low": [max(0.01, value - 0.1) for value in closes],
            "close": closes,
            "volume": [1_000.0] * len(closes),
        },
        index=idx,
    )


class _Report:
    def as_dict(self) -> dict[str, Any]:
        return {"rows_after_cleanup": 3}


def test_policy_and_model_helpers_cover_bad_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert replay._policy_keep_count(0, 0.93, 5) == 0
    assert replay._positive_class_probability_index([object(), "nope"], column_count=2) == 1
    assert replay._positive_class_probability_index(object(), column_count=2) == 1
    assert replay._positive_class_probability_index([0], column_count=1) == 0
    assert replay._extract_symbol_penalties(None) == {}
    assert replay._extract_symbol_penalties({"": {}, "AAPL": "bad"}) == {}

    monkeypatch.setattr(replay, "get_env", lambda *args, **kwargs: "")
    assert replay._resolve_replay_model_path(argparse.Namespace(model_path=None)) is None
    monkeypatch.setattr(replay, "get_env", lambda *args, **kwargs: "relative-model.joblib")
    assert replay._resolve_replay_model_path(argparse.Namespace(model_path=None)).name == "relative-model.joblib"

    missing_model_args = argparse.Namespace(use_model_score=False, model_path=None)
    assert replay._load_replay_model_context(missing_model_args) is None

    model_path = tmp_path / "model.joblib"
    model_path.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(replay, "load_verified_joblib_artifact", lambda path: (_ for _ in ()).throw(ValueError("bad")))
    assert replay._load_replay_model_context(argparse.Namespace(use_model_score=True, model_path=model_path)) is None

    monkeypatch.setattr(replay, "load_verified_joblib_artifact", lambda path: object())
    assert replay._load_replay_model_context(argparse.Namespace(use_model_score=True, model_path=model_path)) is None

    class _ClassesTypeErrorModel:
        classes_ = object()

        def predict_proba(self, values: Any) -> np.ndarray:
            return np.full((len(values), 2), 0.5)

    monkeypatch.setattr(replay, "load_verified_joblib_artifact", lambda path: _ClassesTypeErrorModel())
    context = replay._load_replay_model_context(argparse.Namespace(use_model_score=True, model_path=model_path))
    assert context is not None
    assert context.positive_class_index == 1


def test_rsi_features_profit_and_markout_edge_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    assert replay._safe_rsi(np.asarray([], dtype=float)).size == 0
    monkeypatch.setattr(replay, "rsi_indicator", lambda values, period: (_ for _ in ()).throw(ValueError("boom")))
    assert replay._safe_rsi(np.asarray([1.0, 2.0])).tolist() == [0.0, 0.0]

    duplicate = pd.DataFrame({"close": [1.0, 2.0]}, index=[pd.NaT, pd.NaT])
    sanitized = replay._sanitize_model_feature_index(duplicate, symbol="DUP")
    assert sanitized.index.tolist() == [0, 1]

    bars: list[dict[str, Any]] = []
    replay._attach_policy_context(bars)
    assert bars == []
    bad_ts_bars = [{"symbol": "A", "ts": "bad", "close": 100.0, "score": 0.1, "confidence": 0.8}]
    replay._attach_policy_context(bad_ts_bars)
    assert bad_ts_bars[0]["policy_bar_age_hours"] == 0.0

    assert replay._profit_factor(np.asarray([], dtype=float), np.asarray([], dtype=float)) == 0.0
    assert replay._profit_factor(np.asarray([1.0]), np.asarray([], dtype=float)) is None
    assert replay._max_drawdown_bps([]) == 0.0

    empty_metrics = replay._summarize_markout_fill_metrics(
        fill_events=[
            {"event_type": "accepted"},
            {"event_type": "fill", "client_order_id": ""},
            {"event_type": "fill", "client_order_id": "x"},
            {"event_type": "fill", "client_order_id": "y"},
            {"event_type": "fill", "client_order_id": "z", "fill_price": "bad", "fill_qty": 1},
            {"event_type": "fill", "client_order_id": "w", "fill_price": 0, "fill_qty": 1},
        ],
        order_context_by_client_id={
            "y": {"markout_price": None},
            "z": {"markout_price": 101.0},
            "w": {"markout_price": 101.0},
        },
        fee_bps=1.0,
    )
    assert empty_metrics["samples"] == 0

    win_only = replay._summarize_markout_fill_metrics(
        fill_events=[
            {
                "event_type": "fill",
                "client_order_id": "win",
                "fill_price": 100.0,
                "fill_qty": 2.0,
                "side": "buy",
                "symbol": "WIN",
            }
        ],
        order_context_by_client_id={"win": {"markout_price": 101.0}},
        fee_bps=0.0,
    )
    assert win_only["samples"] == 1
    assert win_only["profit_factor"] is None


def test_model_signal_fallbacks_and_simulate_symbol_max_hold(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _frame([100.0, 101.0, 102.0])

    class _OneDimensionalModel:
        feature_names_in_ = np.asarray(["close", "missing_feature"], dtype=object)

        def predict_proba(self, values: Any) -> np.ndarray:
            return np.asarray([0.2, 0.7, 0.8])

    one_dim_context = replay.ReplayModelContext(
        model=_OneDimensionalModel(),
        model_path="one",
        feature_names=("close", "missing_feature"),
        positive_class_index=0,
        orientation_inverse=False,
        symbol_penalties={},
    )
    score, confidence = replay._compute_model_signal(frame, symbol="ONE", model_context=one_dim_context)
    assert score.iloc[-1] > 0.0
    assert confidence.iloc[0] == pytest.approx(0.8)

    class _BadShapeModel:
        classes_ = np.asarray([0, 1], dtype=int)

        def predict_proba(self, values: Any) -> np.ndarray:
            return np.asarray([[0.1, 0.9]])

    bad_shape_context = replay.ReplayModelContext(
        model=_BadShapeModel(),
        model_path="bad-shape",
        feature_names=("close",),
        positive_class_index=99,
        orientation_inverse=False,
        symbol_penalties={},
    )
    bad_score, bad_confidence = replay._compute_model_signal(frame, symbol="BAD", model_context=bad_shape_context)
    assert bad_score.tolist() == [0.0, 0.0, 0.0]
    assert bad_confidence.tolist() == [0.5, 0.5, 0.5]

    class _RaisingModel:
        def predict_proba(self, values: Any) -> np.ndarray:
            raise ValueError("no score")

    raising_context = replay.ReplayModelContext(
        model=_RaisingModel(),
        model_path="raising",
        feature_names=("close",),
        positive_class_index=0,
        orientation_inverse=False,
        symbol_penalties={},
    )
    fallback_score, fallback_confidence = replay._compute_model_signal(
        frame,
        symbol="FALL",
        model_context=raising_context,
    )
    assert len(fallback_score) == len(frame)
    assert len(fallback_confidence) == len(frame)

    original_compute_signal = replay._compute_signal
    monkeypatch.setattr(
        replay,
        "_compute_signal",
        lambda df: (
            pd.Series([1.0, 1.0, 1.0, 1.0], index=df.index),
            pd.Series([1.0, 1.0, 1.0, 1.0], index=df.index),
        ),
    )
    max_hold_summary = replay._simulate_symbol(
        "MAX",
        _frame([100.0, 101.0, 102.0, 103.0]),
        _cfg(max_hold_bars=2, take_profit_bps=10_000.0),
    )
    monkeypatch.setattr(replay, "_compute_signal", original_compute_signal)
    assert max_hold_summary["trades_detail"][0]["exit_reason"] == "max_hold"

    monkeypatch.setattr(
        replay,
        "_compute_signal",
        lambda df: (
            pd.Series([1.0, 1.0], index=df.index),
            pd.Series([1.0, 1.0], index=df.index),
        ),
    )
    skipped_first = replay._simulate_symbol("SKIP", _frame([0.0, 100.0]), _cfg())
    assert skipped_first["trades"] == 1


def test_input_resolution_normalization_and_run_replay_guard_branches(tmp_path: Path) -> None:
    parser = replay._build_parser()

    with pytest.raises(ValueError, match="empty path"):
        replay._resolve_inputs(argparse.Namespace(csv=Path("."), symbol="", data_dir=None, symbols=""))
    with pytest.raises(ValueError, match="not found"):
        replay._resolve_inputs(argparse.Namespace(csv=tmp_path / "missing.csv", symbol="", data_dir=None, symbols=""))

    (tmp_path / "A.csv").write_text("timestamp,open,high,low,close,volume\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No matching CSV"):
        replay._resolve_inputs(argparse.Namespace(csv=None, data_dir=tmp_path, symbols="B"))

    assert replay._normalize_bar_ts("not-a-date", 7).endswith("00:07:00+00:00")
    assert replay._policy_metric_projection({"aggregate": "bad"}) == {}
    assert replay._profit_factor_delta(None, 1.0) is None
    assert replay._profit_factor_delta("bad", 1.0) is None

    csv_path = tmp_path / "A.csv"
    base = parser.parse_args(["--csv", str(csv_path)])
    base.max_hold_bars = 1
    base.min_hold_bars = 4
    with pytest.raises(ValueError, match="max-hold-bars"):
        replay._run_replay(base)

    for flag, message in [
        ("persist_intents", "requires --simulation-mode"),
        ("policy_sensitivity_mode", "requires --simulation-mode"),
    ]:
        args = parser.parse_args(["--csv", str(csv_path)])
        setattr(args, flag, True)
        with pytest.raises(ValueError, match=message):
            replay._run_replay(args)

    args = parser.parse_args(["--csv", str(csv_path), "--simulation-mode", "--policy-sensitivity-mode", "--persist-intents"])
    with pytest.raises(ValueError, match="cannot be combined"):
        replay._run_replay(args)


def test_parity_simulation_strategy_and_payload_edge_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeReplayEventLoop:
        def __init__(self, *, strategy: Any, **kwargs: Any) -> None:
            self.strategy = strategy

        def run(self, bars: list[dict[str, Any]]) -> dict[str, Any]:
            assert self.strategy({"symbol": "BAD", "score": "bad", "confidence": 1.0}) is None
            assert self.strategy({"symbol": "ZERO", "score": 1.0, "confidence": 1.0, "close": 0.0}) is None
            order = self.strategy(
                {
                    "symbol": "GOOD",
                    "ts": "bad-ts",
                    "score": 1.0,
                    "confidence": 1.0,
                    "close": 100.0,
                    "seq": 5,
                    "next_close": "not-float",
                }
            )
            assert order is not None
            return {
                "events": [],
                "orders": [order],
                "intents": [order],
                "violations": [{"code": "cap"}],
            }

    monkeypatch.setattr(replay, "ReplayEventLoop", _FakeReplayEventLoop)
    monkeypatch.setattr(
        replay,
        "_summarize_markout_fill_metrics",
        lambda **kwargs: {"samples": 0, "per_symbol_samples": []},
    )
    monkeypatch.setattr(
        replay,
        "load_historical_bars",
        lambda path, timestamp_col: (_frame([0.0, 100.0]), _Report()),
    )

    payload = replay._run_parity_simulation(
        args=_args(replay_seed=3),
        cfg=_cfg(),
        symbol_paths={"": tmp_path / "empty.csv"},
    )

    assert payload["aggregate"]["orders_submitted"] == 1
    assert payload["aggregate"]["violations_by_code"] == {"cap": 1}
    assert payload["symbols"][0]["symbol"] == ""


def test_persist_replay_to_oms_handles_skips_and_unfilled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Record:
        def __init__(self, intent_id: str, status: str = "PENDING") -> None:
            self.intent_id = intent_id
            self.status = status

    class _Store:
        database_url = ""

        def __init__(self) -> None:
            self.closed: list[tuple[str, str]] = []
            self.submitted: list[tuple[str, str]] = []
            self.fills: list[tuple[str, float, float | None]] = []

        def get_intent_by_key(self, key: str) -> _Record | None:
            if key.endswith("terminal"):
                return _Record("existing-terminal", "FILLED")
            return None

        def create_intent(self, **kwargs: Any) -> tuple[_Record, bool]:
            return _Record(str(kwargs["intent_id"])), False

        def mark_submitted(self, intent_id: str, broker_order_id: str) -> None:
            self.submitted.append((intent_id, broker_order_id))

        def record_fill(self, intent_id: str, *, fill_qty: float, fill_price: float | None, fill_ts: str | None) -> None:
            self.fills.append((intent_id, fill_qty, fill_price))

        def close_intent(self, intent_id: str, *, final_status: str, last_error: str | None = None) -> None:
            self.closed.append((intent_id, final_status))

    store = _Store()
    monkeypatch.setattr("ai_trading.oms.intent_store.IntentStore", lambda: store)
    monkeypatch.setattr("ai_trading.oms.statuses.is_terminal_intent_status", lambda status: status == "FILLED")

    summary = replay._persist_replay_to_oms(
        replay={
            "intents": [
                "bad",
                {"intent_key": "", "symbol": "A", "qty": 1},
                {"intent_key": "badqty", "symbol": "A", "qty": "bad"},
                {"intent_key": "terminal", "symbol": "A", "qty": 1},
                {"intent_key": "new", "symbol": "A", "side": "buy", "qty": 1, "ts": ""},
                {"intent_key": "partial", "symbol": "B", "side": "buy", "qty": 1},
                {"intent_key": "closed", "symbol": "C", "side": "buy", "qty": 1},
            ],
            "orders": [
                "bad",
                {"client_order_id": "new", "status": "open"},
                {"client_order_id": "partial", "id": "broker-partial"},
                {"client_order_id": "closed", "status": "rejected"},
            ],
            "events": [
                "bad",
                {"event_type": "accepted", "client_order_id": "new"},
                {"event_type": "fill", "client_order_id": "missing", "fill_qty": 1},
                {"event_type": "fill", "client_order_id": "terminal", "fill_qty": 1, "status": "filled"},
                {"event_type": "fill", "client_order_id": "partial", "fill_qty": "bad", "fill_price": "bad", "status": "partial_fill"},
                {"event_type": "fill", "client_order_id": "new", "fill_qty": 1, "fill_price": 100.0, "status": "filled"},
            ],
        },
        prefix="phase2",
    )

    assert summary["created_intents"] == 0
    assert summary["existing_intents"] == 4
    assert summary["existing_terminal_intents_skipped"] == 1
    assert summary["skipped_intents"] == 3
    assert summary["fill_events"] == 2
    assert summary["partially_filled_events"] == 1
    assert summary["filled_terminal_events"] == 1
    assert summary["closed_without_fill"] == 2
    assert len(store.submitted) == 3


def test_run_replay_helper_and_governance_cli_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(replay, "_load_runtime_env_for_replay", lambda args: None)
    monkeypatch.setattr(replay, "_run_replay", lambda args: {"aggregate": {"symbols": 0}})
    assert replay.run_replay(["--csv", str(tmp_path / "x.csv")]) == {"aggregate": {"symbols": 0}}

    assert replay_governance._parse_now("2026-01-02T03:04:05Z").tzinfo == UTC
    assert replay_governance._parse_now("2026-01-02T03:04:05").tzinfo == UTC
    assert replay_governance._parse_now("").tzinfo == UTC

    applied: dict[str, str] = {}
    monkeypatch.setattr(replay_governance, "set_runtime_env_override", lambda key, value: applied.__setitem__(key, value))
    keys = replay_governance._apply_runtime_overrides(
        SimpleNamespace(
            replay_data_dir=None,
            replay_output_dir=None,
            replay_symbols=None,
            replay_timeframes=None,
            replay_start_date=None,
            replay_end_date=None,
            replay_seed=7,
            replay_max_symbol_notional=123.4,
            replay_max_gross_notional=567.8,
            simulate_fills=True,
            enforce_oms_gates=False,
            require_non_regression=None,
            clip_intents_to_caps=True,
        )
    )
    assert "AI_TRADING_REPLAY_SEED" in keys
    assert applied["AI_TRADING_REPLAY_MAX_SYMBOL_NOTIONAL"] == "123.4"
    assert applied["AI_TRADING_REPLAY_ENFORCE_OMS_GATES"] == "0"

    snapshot_path = tmp_path / "replay.json"
    snapshot_path.write_text(
        json.dumps({"violations": "bad", "counterfactual": {"passed": False}}),
        encoding="utf-8",
    )
    assert replay_governance._collect_replay_snapshot(snapshot_path)["violations_count"] == 0

    monkeypatch.setattr(
        replay_governance,
        "run_replay_governance",
        lambda argv: {"status": "ok", "argv": argv, "now": datetime.now(UTC)},
    )
    assert replay_governance.main(["--force"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "ok"
