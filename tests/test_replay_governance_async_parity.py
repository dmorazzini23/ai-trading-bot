from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
import pytest


from ai_trading.core import bot_engine


class _State:
    last_replay_run_date = None


@pytest.fixture(autouse=True)
def _isolate_replay_governance_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_TRADING_REPLAY_REFRESH_FROM_TCA", "0")
    monkeypatch.setenv("AI_TRADING_REPLAY_DECISION_SHADOW_SOURCE_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_REPLAY_REQUIRE_NON_REGRESSION", "0")
    monkeypatch.setenv("AI_TRADING_REPLAY_MAX_SOURCE_AGE_HOURS", "96")


def test_load_replay_bars_requires_timeframe_when_filter_configured(tmp_path: Path) -> None:
    data_path = tmp_path / "replay.jsonl"
    rows = [
        {
            "symbol": "AAPL",
            "ts": "2026-02-18T22:00:00+00:00",
            "close": 189.5,
            "timeframe": "5Min",
        },
        {
            "symbol": "AAPL",
            "ts": "2026-02-18T22:05:00+00:00",
            "close": 190.0,
        },
        {
            "symbol": "AAPL",
            "ts": "2026-02-18T22:10:00+00:00",
            "close": 190.5,
            "timeframe": "1Min",
        },
    ]
    data_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )

    loaded = bot_engine._load_replay_bars(
        path=str(tmp_path),
        symbols={"AAPL"},
        timeframes={"5Min"},
        start_date=None,
        end_date=None,
    )

    assert [row["ts"] for row in loaded] == ["2026-02-18T22:00:00+00:00"]


def test_replay_governance_uses_async_parity_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    now = datetime(2026, 2, 18, 23, 0, tzinfo=UTC)
    monkeypatch.setattr(bot_engine, "_replay_schedule_due", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        bot_engine,
        "_load_replay_bars",
        lambda **_kwargs: [
            {
                "symbol": "AAPL",
                "ts": "2026-02-18T22:00:00+00:00",
                "close": 189.5,
                "side": "buy",
                "qty": 1,
                "client_order_id": "aapl-1",
            },
            {
                "symbol": "AAPL",
                "ts": "2026-02-18T22:01:00+00:00",
                "close": 189.8,
                "side": "buy",
                "qty": 1,
                "client_order_id": "aapl-2",
            },
        ],
    )
    monkeypatch.setenv("AI_TRADING_REPLAY_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("AI_TRADING_REPLAY_SEED", "123")
    monkeypatch.setenv("AI_TRADING_REPLAY_SIMULATE_FILLS", "1")
    monkeypatch.setenv("AI_TRADING_REPLAY_ENFORCE_OMS_GATES", "1")

    state = _State()
    state.effective_policy_hash = "policy-test"
    bot_engine._run_replay_governance(state, now=now, market_open_now=False)

    out_path = tmp_path / "replay_hash_20260218.json"
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "3.0.0"
    assert payload["policy_hash"] == "policy-test"
    assert payload["simulate_fills"] is True
    assert payload["seed"] == 123
    assert payload["orders_submitted"] >= 1
    assert "cap_adjustments_count" in payload
    assert "violations" in payload
    assert isinstance(payload.get("replay_symbol_summary"), dict)
    assert isinstance(payload.get("replay_bucket_summary"), dict)
    assert isinstance(getattr(state, "replay_bucket_summary", None), dict)
    assert state.last_replay_run_date == now.date()


def test_replay_non_regression_failure_writes_current_failed_artifact(
    monkeypatch,
    tmp_path: Path,
) -> None:
    now = datetime(2026, 2, 18, 23, 0, tzinfo=UTC)
    baseline_path = tmp_path / "replay_hash_20260217.json"
    baseline_path.write_text(
        json.dumps(
            {
                "replay_summary": {
                    "sample_count": 10,
                    "net_edge_bps": 5.0,
                    "max_drawdown_pct": 0.01,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(bot_engine, "_replay_schedule_due", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        bot_engine,
        "_load_replay_bars",
        lambda **_kwargs: [
            {
                "symbol": "AAPL",
                "ts": "2026-02-18T22:00:00+00:00",
                "close": 189.5,
                "side": "buy",
                "qty": 1,
                "client_order_id": "aapl-1",
            }
        ],
    )
    monkeypatch.setattr(
        bot_engine,
        "evaluate_counterfactual_non_regression",
        lambda **_kwargs: (False, {"reason": "regression"}),
    )
    monkeypatch.setenv("AI_TRADING_REPLAY_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("AI_TRADING_REPLAY_REQUIRE_NON_REGRESSION", "1")
    monkeypatch.setenv("AI_TRADING_REPLAY_ENFORCE_OMS_GATES", "0")

    state = _State()
    state.effective_policy_hash = "policy-test"
    try:
        bot_engine._run_replay_governance(state, now=now, market_open_now=False)
    except RuntimeError as exc:
        assert str(exc) == "REPLAY_POLICY_NON_REGRESSION_FAILED"
    else:
        raise AssertionError("Expected replay non-regression failure")

    current_path = tmp_path / "replay_hash_20260218.json"
    payload = json.loads(current_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "3.0.0"
    assert payload["policy_hash"] == "policy-test"
    assert payload["counterfactual"]["passed"] is False


def test_replay_governance_enforced_invariants_raise(
    monkeypatch,
) -> None:
    now = datetime(2026, 2, 18, 23, 0, tzinfo=UTC)
    monkeypatch.setattr(bot_engine, "_replay_schedule_due", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        bot_engine,
        "_load_replay_bars",
        lambda **_kwargs: [
            {
                "symbol": "MSFT",
                "ts": "2026-02-18T22:00:00+00:00",
                "close": 405.0,
                "side": "buy",
                "qty": 1,
                "client_order_id": "dup-intent",
            },
            {
                "symbol": "MSFT",
                "ts": "2026-02-18T22:01:00+00:00",
                "close": 405.2,
                "side": "buy",
                "qty": 1,
                "client_order_id": "dup-intent",
            },
        ],
    )
    monkeypatch.setenv("AI_TRADING_REPLAY_SIMULATE_FILLS", "1")
    monkeypatch.setenv("AI_TRADING_REPLAY_ENFORCE_OMS_GATES", "1")

    state = _State()
    try:
        bot_engine._run_replay_governance(state, now=now, market_open_now=False)
    except RuntimeError as exc:
        assert "REPLAY_GOVERNANCE_INVARIANTS_FAILED" in str(exc)
    else:  # pragma: no cover - explicit safety
        raise AssertionError("Expected replay invariants to fail")


def test_replay_governance_resolves_runtime_paths_against_data_dir(
    monkeypatch,
    tmp_path: Path,
) -> None:
    now = datetime(2026, 2, 18, 23, 0, tzinfo=UTC)
    data_root = tmp_path / "data-root"
    data_root.mkdir(parents=True, exist_ok=True)
    captured: dict[str, str] = {}

    monkeypatch.setattr(bot_engine, "_replay_schedule_due", lambda *_args, **_kwargs: True)

    def _load_bars(**kwargs):
        captured["path"] = str(kwargs.get("path", ""))
        return [
            {
                "symbol": "AAPL",
                "ts": "2026-02-18T22:00:00+00:00",
                "close": 189.5,
                "side": "buy",
                "qty": 1,
                "client_order_id": "aapl-1",
            },
        ]

    monkeypatch.setattr(bot_engine, "_load_replay_bars", _load_bars)
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(data_root))
    monkeypatch.setenv("AI_TRADING_REPLAY_DATA_DIR", "runtime/replay_data")
    monkeypatch.setenv("AI_TRADING_REPLAY_OUTPUT_DIR", "runtime/replay_outputs")
    monkeypatch.setenv("AI_TRADING_REPLAY_ENFORCE_OMS_GATES", "0")
    monkeypatch.setenv("AI_TRADING_REPLAY_SIMULATE_FILLS", "1")

    state = _State()
    bot_engine._run_replay_governance(state, now=now, market_open_now=False)

    assert captured["path"] == str(data_root / "runtime" / "replay_data")
    out_path = data_root / "runtime" / "replay_outputs" / "replay_hash_20260218.json"
    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["source_path"] == str(data_root / "runtime" / "replay_data")


def test_replay_governance_defaults_to_tickers_file_universe(
    monkeypatch,
    tmp_path: Path,
) -> None:
    now = datetime(2026, 2, 18, 23, 0, tzinfo=UTC)
    tickers_path = tmp_path / "tickers.csv"
    tickers_path.write_text("AAPL\nMSFT\n", encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.setattr(bot_engine, "_replay_schedule_due", lambda *_args, **_kwargs: True)

    def _load_bars(**kwargs):
        captured["symbols"] = tuple(sorted(kwargs.get("symbols", ())))
        return [
            {
                "symbol": "AAPL",
                "ts": "2026-02-18T22:00:00+00:00",
                "close": 189.5,
                "side": "buy",
                "qty": 1,
                "client_order_id": "aapl-1",
            },
            {
                "symbol": "TSLA",
                "ts": "2026-02-18T22:01:00+00:00",
                "close": 250.0,
                "side": "buy",
                "qty": 1,
                "client_order_id": "tsla-1",
            },
        ]

    monkeypatch.setattr(bot_engine, "_load_replay_bars", _load_bars)
    monkeypatch.setenv("AI_TRADING_TICKERS_FILE", str(tickers_path))
    monkeypatch.setenv("AI_TRADING_REPLAY_SYMBOLS", "")
    monkeypatch.setenv("AI_TRADING_CANARY_SYMBOLS", "")
    monkeypatch.setenv("AI_TRADING_SYMBOLS", "")
    monkeypatch.setenv("AI_TRADING_REPLAY_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("AI_TRADING_REPLAY_ENFORCE_OMS_GATES", "0")
    monkeypatch.setenv("AI_TRADING_REPLAY_SIMULATE_FILLS", "1")

    state = _State()
    bot_engine._run_replay_governance(state, now=now, market_open_now=False, force=True)

    assert captured["symbols"] == ("AAPL", "MSFT")


def test_replay_governance_defaults_to_canary_universe_before_tickers_file(
    monkeypatch,
    tmp_path: Path,
) -> None:
    now = datetime(2026, 2, 18, 23, 0, tzinfo=UTC)
    tickers_path = tmp_path / "tickers.csv"
    tickers_path.write_text("AAPL\nMSFT\n", encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.setattr(bot_engine, "_replay_schedule_due", lambda *_args, **_kwargs: True)

    def _load_bars(**kwargs):
        captured["symbols"] = tuple(sorted(kwargs.get("symbols", ())))
        return [
            {
                "symbol": "AMZN",
                "ts": "2026-02-18T22:00:00+00:00",
                "close": 189.5,
                "side": "buy",
                "qty": 1,
                "client_order_id": "amzn-1",
            }
        ]

    monkeypatch.setattr(bot_engine, "_load_replay_bars", _load_bars)
    monkeypatch.setenv("AI_TRADING_TICKERS_FILE", str(tickers_path))
    monkeypatch.setenv("AI_TRADING_REPLAY_SYMBOLS", "")
    monkeypatch.setenv("AI_TRADING_CANARY_SYMBOLS", "AAPL,AMZN")
    monkeypatch.setenv("AI_TRADING_SYMBOLS", "SPY")
    monkeypatch.setenv("AI_TRADING_REPLAY_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("AI_TRADING_REPLAY_ENFORCE_OMS_GATES", "0")
    monkeypatch.setenv("AI_TRADING_REPLAY_SIMULATE_FILLS", "1")

    state = _State()
    bot_engine._run_replay_governance(state, now=now, market_open_now=False, force=True)

    assert captured["symbols"] == ("AAPL", "AMZN")


def test_replay_governance_filters_canary_universe_to_paper_sampling_allowlist(
    monkeypatch,
    tmp_path: Path,
) -> None:
    now = datetime(2026, 2, 18, 23, 0, tzinfo=UTC)
    captured: dict[str, object] = {}

    monkeypatch.setattr(bot_engine, "_replay_schedule_due", lambda *_args, **_kwargs: True)

    def _load_bars(**kwargs):
        captured["symbols"] = tuple(sorted(kwargs.get("symbols", ())))
        return [
            {
                "symbol": "AAPL",
                "ts": "2026-02-18T22:00:00+00:00",
                "close": 189.5,
                "side": "buy",
                "qty": 1,
                "client_order_id": "aapl-1",
            }
        ]

    monkeypatch.setattr(bot_engine, "_load_replay_bars", _load_bars)
    monkeypatch.setenv("AI_TRADING_REPLAY_SYMBOLS", "")
    monkeypatch.setenv("AI_TRADING_CANARY_SYMBOLS", "AAPL,AMZN,MSFT")
    monkeypatch.setenv("AI_TRADING_PAPER_SAMPLING_ALLOWED_SYMBOLS", "AAPL,AMZN")
    monkeypatch.setenv("AI_TRADING_SYMBOLS", "SPY")
    monkeypatch.setenv("AI_TRADING_REPLAY_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("AI_TRADING_REPLAY_ENFORCE_OMS_GATES", "0")
    monkeypatch.setenv("AI_TRADING_REPLAY_SIMULATE_FILLS", "1")

    state = _State()
    bot_engine._run_replay_governance(state, now=now, market_open_now=False, force=True)

    assert captured["symbols"] == ("AAPL", "AMZN")


def test_replay_governance_force_bypasses_schedule_gate(
    monkeypatch,
    tmp_path: Path,
) -> None:
    now = datetime(2026, 2, 18, 23, 0, tzinfo=UTC)
    monkeypatch.setattr(bot_engine, "_replay_schedule_due", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        bot_engine,
        "_load_replay_bars",
        lambda **_kwargs: [
            {
                "symbol": "AAPL",
                "ts": "2026-02-18T22:00:00+00:00",
                "close": 189.5,
                "side": "buy",
                "qty": 1,
                "client_order_id": "aapl-1",
            }
        ],
    )
    monkeypatch.setenv("AI_TRADING_REPLAY_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("AI_TRADING_REPLAY_SIMULATE_FILLS", "1")
    monkeypatch.setenv("AI_TRADING_REPLAY_ENFORCE_OMS_GATES", "0")

    state = _State()
    bot_engine._run_replay_governance(state, now=now, market_open_now=False, force=True)

    out_path = tmp_path / "replay_hash_20260218.json"
    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["rows"] == 1
    assert state.last_replay_run_date == now.date()


def test_replay_symbol_summary_metrics_aggregates_by_symbol() -> None:
    result = {
        "orders": [
            {"id": "o1", "symbol": "AAPL", "side": "buy", "limit_price": 100.0},
            {"id": "o2", "symbol": "AAPL", "side": "sell", "limit_price": 100.0},
            {"id": "o3", "symbol": "MSFT", "side": "buy", "limit_price": 200.0},
        ],
        "events": [
            {
                "event_type": "fill",
                "order_id": "o1",
                "symbol": "AAPL",
                "side": "buy",
                "fill_price": 99.0,
            },
            {
                "event_type": "fill",
                "order_id": "o2",
                "symbol": "AAPL",
                "side": "sell",
                "fill_price": 101.0,
            },
            {
                "event_type": "fill",
                "order_id": "o3",
                "symbol": "MSFT",
                "side": "buy",
                "fill_price": 202.0,
            },
        ],
    }

    summary = bot_engine._replay_symbol_summary_metrics(result)
    assert int(summary["AAPL"]["sample_count"]) == 2
    assert float(summary["AAPL"]["net_edge_bps"]) > 0.0
    assert float(summary["AAPL"]["win_rate"]) == 1.0
    assert int(summary["MSFT"]["sample_count"]) == 1
    assert float(summary["MSFT"]["net_edge_bps"]) < 0.0


def test_load_latest_replay_symbol_summary_reads_recent_file(tmp_path: Path) -> None:
    payload = {
        "ts": datetime(2026, 2, 18, 23, 0, tzinfo=UTC).isoformat(),
        "replay_symbol_summary": {
            "AAPL": {
                "sample_count": 24,
                "net_edge_bps": 3.1,
                "win_rate": 0.62,
                "profit_factor": 1.4,
            }
        },
    }
    path = tmp_path / "replay_hash_20260218.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    summary, context = bot_engine._load_latest_replay_symbol_summary(
        tmp_path,
        max_age_hours=365 * 24,
    )
    assert "AAPL" in summary
    assert int(summary["AAPL"]["sample_count"]) == 24
    assert float(summary["AAPL"]["net_edge_bps"]) == 3.1
    assert context.get("path") == str(path)


def test_replay_bucket_summary_metrics_aggregates_session_regime() -> None:
    result = {
        "orders": [
            {
                "id": "o1",
                "symbol": "AAPL",
                "side": "buy",
                "limit_price": 100.0,
                "client_order_id": "cid-1",
            },
            {
                "id": "o2",
                "symbol": "AAPL",
                "side": "sell",
                "limit_price": 100.0,
                "client_order_id": "cid-2",
            },
        ],
        "events": [
            {
                "event_type": "fill",
                "order_id": "o1",
                "client_order_id": "cid-1",
                "symbol": "AAPL",
                "side": "buy",
                "fill_price": 99.0,
                "ts": "2026-02-18T14:31:00+00:00",
            },
            {
                "event_type": "fill",
                "order_id": "o2",
                "client_order_id": "cid-2",
                "symbol": "AAPL",
                "side": "sell",
                "fill_price": 101.0,
                "ts": "2026-02-18T14:32:00+00:00",
            },
        ],
    }
    context = {
        "cid-1": {"session_token": "opening", "regime_token": "risk_on"},
        "cid-2": {"session_token": "opening", "regime_token": "risk_on"},
    }
    summary = bot_engine._replay_bucket_summary_metrics(
        result,
        order_context_by_client_id=context,
    )
    by_session = summary["by_symbol_session"]
    by_session_regime = summary["by_symbol_session_regime"]
    assert "AAPL:opening" in by_session
    assert int(by_session["AAPL:opening"]["sample_count"]) == 2
    assert "AAPL:opening:risk_on" in by_session_regime
    assert int(by_session_regime["AAPL:opening:risk_on"]["sample_count"]) == 2


def test_load_latest_replay_quality_summaries_reads_bucket_payload(
    tmp_path: Path,
) -> None:
    payload = {
        "ts": datetime(2026, 2, 18, 23, 0, tzinfo=UTC).isoformat(),
        "replay_symbol_summary": {
            "AAPL": {"sample_count": 12, "net_edge_bps": 1.2, "win_rate": 0.55}
        },
        "replay_bucket_summary": {
            "by_symbol_session": {
                "AAPL:opening": {
                    "sample_count": 12,
                    "net_edge_bps": 1.4,
                    "win_rate": 0.58,
                }
            },
            "by_symbol_session_regime": {
                "AAPL:opening:risk_on": {
                    "sample_count": 8,
                    "net_edge_bps": 1.8,
                    "win_rate": 0.62,
                }
            },
        },
    }
    path = tmp_path / "replay_hash_20260218.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    symbol_summary, session_summary, session_regime_summary, context = (
        bot_engine._load_latest_replay_quality_summaries(
            tmp_path,
            max_age_hours=365 * 24,
        )
    )
    assert "AAPL" in symbol_summary
    assert "AAPL:opening" in session_summary
    assert "AAPL:opening:risk_on" in session_regime_summary
    assert context.get("path") == str(path)


def test_refresh_replay_dataset_from_tca_deduplicates_recent_decisions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    now = datetime(2026, 2, 18, 23, 0, tzinfo=UTC)
    tca_path = tmp_path / "tca_records.jsonl"
    recent = {
        "symbol": "AAPL",
        "side": "buy",
        "requested_qty": 2,
        "submit_price_reference": 190.0,
        "order_type": "limit",
        "benchmark": {"submit_ts": "2026-02-18T20:00:00+00:00"},
    }
    stale = {
        **recent,
        "benchmark": {"submit_ts": "2025-12-01T20:00:00+00:00"},
    }
    tca_path.write_text(
        "\n".join((json.dumps(recent), json.dumps(recent), json.dumps(stale))) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AI_TRADING_REPLAY_REFRESH_FROM_TCA", "1")
    monkeypatch.setenv("AI_TRADING_TCA_PATH", str(tca_path))
    monkeypatch.setenv("AI_TRADING_REPLAY_TCA_LOOKBACK_DAYS", "30")

    output_path, context = bot_engine._refresh_replay_dataset_from_tca(
        data_dir=tmp_path / "replay_data",
        now=now,
    )

    assert output_path is not None
    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["qty"] == 2.0
    assert rows[0]["source_kind"] == "tca_decision"
    assert rows[0]["client_order_id"]
    assert context["decision_rows"] == 1


def test_refresh_replay_dataset_accepts_only_valid_parity_shadow_decisions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    now = datetime(2026, 2, 18, 23, 0, tzinfo=UTC)
    tca_path = tmp_path / "tca_records.jsonl"
    decision_path = tmp_path / "decision_records.jsonl"
    tca_path.write_text("", encoding="utf-8")
    valid = {
        "gates": ["REPLAY_LIVE_PARITY_GATE_FAILED"],
        "decision_journal": {
            "submitted": False,
            "bar_ts": "2026-02-18T20:00:00+00:00",
            "reasons": ["REPLAY_LIVE_PARITY_GATE_FAILED"],
            "order_intent": {
                "symbol": "MSFT",
                "side": "sell",
                "qty": 3,
                "limit_price": 405.25,
                "client_order_id": "shadow-msft-1",
            },
        },
    }
    submitted = json.loads(json.dumps(valid))
    submitted["decision_journal"]["submitted"] = True
    invalid_side = json.loads(json.dumps(valid))
    invalid_side["decision_journal"]["order_intent"]["side"] = "sell_short"
    invalid_qty = json.loads(json.dumps(valid))
    invalid_qty["decision_journal"]["order_intent"]["qty"] = 0
    unmarked = json.loads(json.dumps(valid))
    unmarked["gates"] = []
    unmarked["decision_journal"]["reasons"] = []
    decision_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (valid, submitted, invalid_side, invalid_qty, unmarked)
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AI_TRADING_REPLAY_REFRESH_FROM_TCA", "1")
    monkeypatch.setenv("AI_TRADING_TCA_PATH", str(tca_path))
    monkeypatch.setenv("AI_TRADING_REPLAY_DECISION_SHADOW_SOURCE_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_DECISION_LOG_PATH", str(decision_path))

    output_path, context = bot_engine._refresh_replay_dataset_from_tca(
        data_dir=tmp_path / "replay_data",
        now=now,
    )

    assert output_path is not None
    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert rows == [
        {
            "client_order_id": "shadow-msft-1",
            "close": 405.25,
            "order_type": "limit",
            "qty": 3.0,
            "regime_profile": "unknown",
            "side": "sell",
            "source_kind": "decision_journal_parity_shadow",
            "symbol": "MSFT",
            "timeframe": "5Min",
            "ts": "2026-02-18T20:00:00+00:00",
        }
    ]
    assert context["shadow_scanned_records"] == 5
    assert context["shadow_rejected_records"] == 4
    assert context["shadow_accepted_records"] == 1
    assert context["shadow_decision_rows"] == 1
    assert context["shadow_source_path"] == str(decision_path)
    assert set(context["source_paths"]) == {str(tca_path), str(decision_path)}


def test_refresh_replay_dataset_prefers_tca_on_shadow_identity_collision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    now = datetime(2026, 2, 18, 23, 0, tzinfo=UTC)
    tca_path = tmp_path / "tca_records.jsonl"
    decision_path = tmp_path / "decision_records.jsonl"
    tca_path.write_text(
        json.dumps(
            {
                "symbol": "AAPL",
                "side": "buy",
                "requested_qty": 2,
                "submit_price_reference": 190.0,
                "client_order_id": "shared-order-id",
                "benchmark": {"submit_ts": "2026-02-18T20:00:00+00:00"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    decision_path.write_text(
        json.dumps(
            {
                "gates": ["REPLAY_LIVE_PARITY_GATE_FAILED"],
                "decision_journal": {
                    "submitted": False,
                    "bar_ts": "2026-02-18T20:00:00+00:00",
                    "order_intent": {
                        "symbol": "AAPL",
                        "side": "buy",
                        "qty": 7,
                        "limit_price": 195.0,
                        "client_order_id": "shared-order-id",
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AI_TRADING_REPLAY_REFRESH_FROM_TCA", "1")
    monkeypatch.setenv("AI_TRADING_TCA_PATH", str(tca_path))
    monkeypatch.setenv("AI_TRADING_REPLAY_DECISION_SHADOW_SOURCE_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_DECISION_LOG_PATH", str(decision_path))

    output_path, context = bot_engine._refresh_replay_dataset_from_tca(
        data_dir=tmp_path / "replay_data",
        now=now,
    )

    assert output_path is not None
    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["source_kind"] == "tca_decision"
    assert rows[0]["qty"] == 2.0
    assert rows[0]["close"] == 190.0
    assert context["shadow_accepted_records"] == 1
    assert context["shadow_decision_rows"] == 0
    assert context["tca_decision_rows"] == 1


def test_rollout_replay_eligibility_fails_closed_for_required_live_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "AI_TRADING_RUNTIME_GONOGO_REQUIRE_REPLAY_LIVE_PARITY_GATE",
        "1",
    )
    state = _State()

    assert bot_engine._rollout_replay_advance_eligibility(
        state=state,
        execution_mode="live",
    ) == (False, "replay_live_parity_gate_absent")

    for gate, reason in (
        ({"enabled": False, "available": True, "ok": True}, "replay_live_parity_gate_disabled"),
        ({"enabled": True, "available": False, "ok": True}, "replay_live_parity_gate_unavailable"),
        (
            {
                "enabled": True,
                "available": True,
                "ok": False,
                "reason": "counterfactual_failed",
            },
            "counterfactual_failed",
        ),
    ):
        state.replay_live_parity_gate = gate
        assert bot_engine._rollout_replay_advance_eligibility(
            state=state,
            execution_mode="live",
        ) == (False, reason)

    state.replay_live_parity_gate = {
        "enabled": True,
        "available": True,
        "ok": True,
    }
    assert bot_engine._rollout_replay_advance_eligibility(
        state=state,
        execution_mode="live",
    ) == (True, "")


def test_rollout_replay_eligibility_does_not_block_paper_or_optional_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _State()
    monkeypatch.setenv(
        "AI_TRADING_RUNTIME_GONOGO_REQUIRE_REPLAY_LIVE_PARITY_GATE",
        "1",
    )
    assert bot_engine._rollout_replay_advance_eligibility(
        state=state,
        execution_mode="paper",
    ) == (True, "")

    monkeypatch.setenv(
        "AI_TRADING_RUNTIME_GONOGO_REQUIRE_REPLAY_LIVE_PARITY_GATE",
        "0",
    )
    assert bot_engine._rollout_replay_advance_eligibility(
        state=state,
        execution_mode="live",
    ) == (True, "")


def test_replay_governance_preserves_zero_quantity_observations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    now = datetime(2026, 2, 18, 23, 0, tzinfo=UTC)
    monkeypatch.setattr(bot_engine, "_replay_schedule_due", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        bot_engine,
        "_load_replay_bars",
        lambda **_kwargs: [
            {
                "symbol": "AAPL",
                "ts": "2026-02-18T22:00:00+00:00",
                "close": 189.5,
                "side": "buy",
                "qty": 0,
            },
            {
                "symbol": "AAPL",
                "ts": "2026-02-18T22:05:00+00:00",
                "close": 190.0,
                "side": "buy",
                "qty": 1,
                "client_order_id": "real-decision",
            },
        ],
    )
    monkeypatch.setenv("AI_TRADING_REPLAY_OUTPUT_DIR", str(tmp_path))

    state = _State()
    state.effective_policy_hash = "policy-test"
    bot_engine._run_replay_governance(state, now=now, market_open_now=False)

    payload = json.loads(
        (tmp_path / "replay_hash_20260218.json").read_text(encoding="utf-8")
    )
    assert payload["rows"] == 2
    assert payload["orders_submitted"] == 1
    assert payload["comparison"]["baseline_orders_submitted"] == 1


def test_replay_governance_writes_artifact_then_blocks_stale_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    now = datetime(2026, 2, 18, 23, 0, tzinfo=UTC)
    monkeypatch.setattr(bot_engine, "_replay_schedule_due", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        bot_engine,
        "_load_replay_bars",
        lambda **_kwargs: [
            {
                "symbol": "AAPL",
                "ts": "2026-02-01T22:00:00+00:00",
                "close": 189.5,
                "side": "buy",
                "qty": 1,
                "client_order_id": "stale-1",
            },
            {
                "symbol": "AAPL",
                "ts": "2026-02-01T22:05:00+00:00",
                "close": 190.0,
                "side": "sell",
                "qty": 1,
                "client_order_id": "stale-2",
            },
        ],
    )
    monkeypatch.setenv("AI_TRADING_REPLAY_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("AI_TRADING_REPLAY_MAX_SOURCE_AGE_HOURS", "96")

    state = _State()
    state.effective_policy_hash = "policy-test"
    with pytest.raises(RuntimeError, match="REPLAY_SOURCE_DATA_STALE"):
        bot_engine._run_replay_governance(state, now=now, market_open_now=False)

    payload = json.loads(
        (tmp_path / "replay_hash_20260218.json").read_text(encoding="utf-8")
    )
    assert payload["source_data"]["fresh"] is False
    assert payload["source_data"]["age_hours"] > 96.0
