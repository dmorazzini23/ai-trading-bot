from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from ai_trading.core import bot_engine


class _State:
    last_replay_run_date = None


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
    bot_engine._run_replay_governance(state, now=now, market_open_now=False)

    out_path = tmp_path / "replay_hash_20260218.json"
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["simulate_fills"] is True
    assert payload["seed"] == 123
    assert payload["orders_submitted"] >= 1
    assert "cap_adjustments_count" in payload
    assert "violations" in payload
    assert isinstance(payload.get("replay_symbol_summary"), dict)
    assert isinstance(payload.get("replay_bucket_summary"), dict)
    assert isinstance(getattr(state, "replay_bucket_summary", None), dict)
    assert state.last_replay_run_date == now.date()


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
