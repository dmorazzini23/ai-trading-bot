"""EOD summary gating, same-day P&L derivation and incident flap debouncing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools import mcp_slack_alerts_server as slack_srv


def _eod_snapshot(**overrides: Any) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "report_date": "2026-10-16",
        "go_no_go_gate_passed": False,
        "go_no_go_failed_checks": ["required_model_stale"],
        "net_pnl": 0.76,
        "pnl_basis": "same_day_fills",
        "profit_factor": 1.1,
        "win_rate": 0.75,
        "closed_trades": 4,
        "execution_capture_ratio": 0.32,
        "slippage_drag_bps": 8.4,
        "order_reject_rate_pct": None,
        "open_position_reconciliation_mismatch_count": 0,
        "open_position_reconciliation_max_abs_delta_qty": 0.0,
        "health_status": "degraded",
        "health_reason": "required_model_stale",
        "provider_status": "warming_up",
        "provider_reason": "market_closed",
        "provider_active": "alpaca",
        "using_backup": False,
        "broker_status": "connected",
        "top_loss_symbols": [],
        "top_loss_basis": "same_day_fills",
        "timestamp": "2026-10-16T20:05:00Z",
        "learning": {},
    }
    snapshot.update(overrides)
    return snapshot


def _stub_training_gate(monkeypatch) -> None:
    monkeypatch.setattr(
        slack_srv,
        "_after_hours_training_gate",
        lambda report_date, args: {"ready": True, "reason": "not_required"},
    )
    monkeypatch.setattr(slack_srv, "_block_eod_on_training_gate", lambda args: False)


def test_eod_summary_eligible_after_close_while_health_reason_is_stale_model(monkeypatch) -> None:
    _stub_training_gate(monkeypatch)
    monkeypatch.setattr(slack_srv, "_collect_eod_summary_snapshot", lambda args: _eod_snapshot())

    result = slack_srv.tool_runtime_eod_summary_snapshot({"require_market_closed": True})

    assert result["market_closed"] is True
    assert result["eligible"] is True


def test_eod_summary_not_eligible_during_session(monkeypatch) -> None:
    _stub_training_gate(monkeypatch)
    monkeypatch.setattr(
        slack_srv,
        "_collect_eod_summary_snapshot",
        lambda args: _eod_snapshot(provider_status="healthy", provider_reason="healthy"),
    )

    result = slack_srv.tool_runtime_eod_summary_snapshot({"require_market_closed": True})

    assert result["market_closed"] is False
    assert result["eligible"] is False


def test_notify_eod_summary_sends_after_close_despite_stale_model_reason(
    monkeypatch, tmp_path: Path
) -> None:
    _stub_training_gate(monkeypatch)
    monkeypatch.setattr(slack_srv, "_collect_eod_summary_snapshot", lambda args: _eod_snapshot())
    posts: list[dict[str, Any]] = []

    def _fake_post(webhook_url: str, payload: dict[str, Any], timeout_s: float = 5.0) -> int:
        posts.append(payload)
        return 200

    monkeypatch.setattr(slack_srv, "_post_slack_message", _fake_post)
    args = {
        "state_path": str(tmp_path / "eod_state.json"),
        "webhook_url": "https://hooks.slack.test/example",
        "require_market_closed": True,
    }

    result = slack_srv.tool_notify_eod_summary(args)

    assert result["sent"] is True
    assert len(posts) == 1


def test_notify_eod_summary_blocks_during_session(monkeypatch, tmp_path: Path) -> None:
    _stub_training_gate(monkeypatch)
    monkeypatch.setattr(
        slack_srv,
        "_collect_eod_summary_snapshot",
        lambda args: _eod_snapshot(provider_status="healthy", provider_reason="healthy"),
    )
    monkeypatch.setattr(slack_srv, "_post_slack_message", lambda *a, **k: 200)

    result = slack_srv.tool_notify_eod_summary(
        {
            "state_path": str(tmp_path / "eod_state.json"),
            "webhook_url": "https://hooks.slack.test/example",
            "require_market_closed": True,
        }
    )

    assert result["sent"] is False
    assert result["reason"] == "market_not_closed"


def test_same_day_fill_summary_derives_every_trading_field_from_report_date_fills(
    monkeypatch, tmp_path: Path
) -> None:
    rows = [
        # Previous session: must be ignored.
        {"ts": "2026-10-15T15:00:00+00:00", "symbol": "NVDA", "side": "buy", "fill_qty": 1, "fill_price": 10.0},
        {"ts": "2026-10-16T14:00:00+00:00", "symbol": "AAPL", "side": "buy", "fill_qty": 2, "fill_price": 100.0},
        {"ts": "2026-10-16T15:00:00+00:00", "symbol": "AAPL", "side": "sell", "fill_qty": 2, "fill_price": 101.0},
        {"ts": "2026-10-16T15:30:00+00:00", "symbol": "MSFT", "side": "buy", "fill_qty": 1, "fill_price": 50.0},
        # 00:30 UTC on the 17th is 20:30 Eastern on the 16th: same session.
        {"ts": "2026-10-17T00:30:00+00:00", "symbol": "MSFT", "side": "sell", "fill_qty": 1, "fill_price": 48.0},
    ]
    path = tmp_path / "fill_events.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setenv("AI_TRADING_FILL_EVENTS_PATH", str(path))

    summary = slack_srv._same_day_fill_round_trip_summary({}, "2026-10-16")

    assert summary["status"] == "ready"
    assert summary["same_day_fill_events"] == 4
    assert summary["closed_trades"] == 2
    assert summary["winning_trades"] == 1
    assert summary["win_rate"] == 0.5
    assert summary["net_pnl"] == 0.0
    assert summary["gross_profit"] == 2.0
    assert summary["gross_loss"] == 2.0
    assert summary["profit_factor"] == 1.0
    assert summary["top_loss_symbols"] == [{"symbol": "MSFT", "net_pnl": -2.0}]
    assert summary["pnl_by_symbol"] == {"AAPL": 2.0, "MSFT": -2.0}


def test_eod_snapshot_prefers_same_day_fill_fields_over_cumulative_trade_history(
    monkeypatch,
) -> None:
    report = {
        "go_no_go": {
            "gate_passed": False,
            "failed_checks": ["required_model_stale"],
            "observed": {"trade_metric_scope": "all_time", "net_pnl": -3.94, "closed_trades": 1, "win_rate": 0.0},
        },
        "trade_history": {
            "top_loss_drivers": {
                "symbols": [
                    {"name": "META", "net_pnl": -122.94},
                    {"name": "QQQ", "net_pnl": -75.18},
                    {"name": "AMD", "net_pnl": -35.53},
                ]
            },
            "daily": [
                {"date": "2026-10-16", "net_pnl": -3.94, "trades": 1, "win_rate": 0.0, "profit_factor": 0.0}
            ],
        },
    }
    monkeypatch.setattr(slack_srv, "_runtime_report_payload", lambda: report)
    monkeypatch.setattr(slack_srv, "_normalized_runtime_report_payload", lambda payload: payload)
    monkeypatch.setattr(slack_srv, "_extract_report_date", lambda payload: "2026-10-16")
    monkeypatch.setattr(
        slack_srv,
        "_safe_health_payload",
        lambda **kwargs: {
            "status": "degraded",
            "reason": "required_model_stale",
            "data_provider": {"status": "warming_up", "reason": "market_closed", "active": "alpaca"},
            "broker": {"status": "connected"},
            "timestamp": "2026-10-16T20:05:00Z",
        },
    )
    monkeypatch.setattr(slack_srv, "_collect_learning_snapshot", lambda args, health: {})
    same_day = {
        "status": "ready",
        "closed_trades": 4,
        "winning_trades": 3,
        "win_rate": 0.75,
        "profit_factor": 1.6,
        "net_pnl": 0.76,
        "top_loss_symbols": [{"symbol": "AMZN", "net_pnl": -1.10}],
        "same_day_fill_events": 7,
    }
    monkeypatch.setattr(slack_srv, "_same_day_fill_round_trip_summary", lambda args, report_date: same_day)

    snapshot = slack_srv._collect_eod_summary_snapshot({})

    assert snapshot["report_date"] == "2026-10-16"
    assert snapshot["pnl_basis"] == "same_day_fills"
    assert snapshot["net_pnl"] == 0.76
    assert snapshot["closed_trades"] == 4
    assert snapshot["win_rate"] == 0.75
    assert snapshot["profit_factor"] == 1.6
    assert snapshot["top_loss_symbols"] == [{"symbol": "AMZN", "net_pnl": -1.10}]
    assert snapshot["top_loss_basis"] == "same_day_fills"
    assert snapshot["provider_reason"] == "market_closed"
    assert snapshot["trade_history"]["top_loss_symbols"][0]["symbol"] == "META"

    payload = slack_srv._eod_message_payload(snapshot)
    rendered = json.dumps(payload)
    assert "AMZN" in rendered
    assert "META" not in rendered


def test_eod_snapshot_never_shows_cumulative_losers_without_same_day_fills(monkeypatch) -> None:
    report = {
        "go_no_go": {"gate_passed": True, "failed_checks": [], "observed": {}},
        "trade_history": {
            "top_loss_drivers": {"symbols": [{"name": "META", "net_pnl": -122.94}]},
            "daily": [{"date": "2026-10-16", "net_pnl": -3.94, "trades": 1}],
        },
    }
    monkeypatch.setattr(slack_srv, "_runtime_report_payload", lambda: report)
    monkeypatch.setattr(slack_srv, "_normalized_runtime_report_payload", lambda payload: payload)
    monkeypatch.setattr(slack_srv, "_extract_report_date", lambda payload: "2026-10-16")
    monkeypatch.setattr(
        slack_srv,
        "_safe_health_payload",
        lambda **kwargs: {"status": "healthy", "reason": "market_closed", "data_provider": {}, "broker": {}},
    )
    monkeypatch.setattr(slack_srv, "_collect_learning_snapshot", lambda args, health: {})
    monkeypatch.setattr(
        slack_srv,
        "_same_day_fill_round_trip_summary",
        lambda args, report_date: {"status": "no_same_day_fills", "report_date": report_date},
    )

    snapshot = slack_srv._collect_eod_summary_snapshot({})

    assert snapshot["top_loss_symbols"] == []
    assert snapshot["top_loss_basis"] == "unavailable"
    assert snapshot["trade_history"]["top_loss_symbols"][0]["symbol"] == "META"


def _incident_snapshot(*, broker_unknown: bool) -> dict[str, Any]:
    """A degraded-health incident; a broker blip adds material go/no-go triggers."""

    snapshot: dict[str, Any] = {
        "go_no_go_gate_passed": True,
        "go_no_go_failed_checks": [],
        "execution_capture_ratio": 0.2,
        "slippage_drag_bps": 7.0,
        "health_ok": False,
        "health_status": "degraded",
        "health_reason": "runtime_gate_failed",
        "provider_status": "healthy",
        "provider_active": "alpaca",
        "provider_reason": "runtime_gate_failed",
        "using_backup": False,
        "broker_status": "connected",
        "timestamp": "2026-10-16T15:00:00Z",
    }
    if broker_unknown:
        snapshot.update(
            {
                "broker_status": "unknown",
                "go_no_go_gate_passed": False,
                "go_no_go_failed_checks": ["broker_snapshot_stale"],
            }
        )
    return snapshot


def _run_notify_sequence(monkeypatch, tmp_path: Path, snapshots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    posts: list[dict[str, Any]] = []
    calls = {"n": 0}

    def _fake_collect(_args: dict[str, Any]) -> dict[str, Any]:
        snapshot = snapshots[min(calls["n"], len(snapshots) - 1)]
        calls["n"] += 1
        return snapshot

    def _fake_post(webhook_url: str, payload: dict[str, Any], timeout_s: float = 5.0) -> int:
        posts.append(payload)
        return 200

    monkeypatch.setattr(slack_srv, "_collect_runtime_snapshot", _fake_collect)
    monkeypatch.setattr(slack_srv, "_post_slack_message", _fake_post)
    args = {
        "state_path": str(tmp_path / "slack_state.json"),
        "webhook_url": "https://hooks.slack.test/example",
    }
    results = [slack_srv.tool_notify_incident_channel(args) for _ in snapshots]
    assert len(posts) == sum(1 for result in results if result["sent"])
    return results


def test_one_cycle_broker_blip_does_not_re_alert(monkeypatch, tmp_path: Path) -> None:
    results = _run_notify_sequence(
        monkeypatch,
        tmp_path,
        [
            _incident_snapshot(broker_unknown=False),
            _incident_snapshot(broker_unknown=True),
            _incident_snapshot(broker_unknown=False),
            _incident_snapshot(broker_unknown=True),
            _incident_snapshot(broker_unknown=False),
        ],
    )

    assert [result["sent"] for result in results] == [True, False, False, False, False]
    assert results[1]["reason"] == "trigger_change_confirmation_pending"
    assert results[1]["pending_trigger_set_count"] == 1
    assert results[2]["reason"] == "repeat_cooldown_active"
    assert results[3]["reason"] == "trigger_change_confirmation_pending"
    assert results[3]["pending_trigger_set_count"] == 1
    state = json.loads((tmp_path / "slack_state.json").read_text())
    assert "pending_trigger_set" not in state


def test_persistent_trigger_change_still_alerts(monkeypatch, tmp_path: Path) -> None:
    results = _run_notify_sequence(
        monkeypatch,
        tmp_path,
        [
            _incident_snapshot(broker_unknown=False),
            _incident_snapshot(broker_unknown=True),
            _incident_snapshot(broker_unknown=True),
        ],
    )

    assert [result["sent"] for result in results] == [True, False, True]
    assert results[1]["reason"] == "trigger_change_confirmation_pending"
    assert "go_no_go_failed" in results[2]["triggers"]


def test_genuine_broker_outage_alerts_immediately(monkeypatch, tmp_path: Path) -> None:
    outage = _incident_snapshot(broker_unknown=False)
    outage.update({"broker_status": "disconnected"})
    results = _run_notify_sequence(
        monkeypatch,
        tmp_path,
        [_incident_snapshot(broker_unknown=False), outage],
    )

    assert [result["sent"] for result in results] == [True, True]
    assert "broker_disconnected" in results[1]["triggers"]


def test_first_incident_is_not_delayed_by_confirmation(monkeypatch, tmp_path: Path) -> None:
    results = _run_notify_sequence(monkeypatch, tmp_path, [_incident_snapshot(broker_unknown=True)])

    assert results[0]["sent"] is True


def test_trigger_change_confirmations_can_be_disabled(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AI_TRADING_SLACK_TRIGGER_CHANGE_CONFIRMATIONS", "1")
    results = _run_notify_sequence(
        monkeypatch,
        tmp_path,
        [_incident_snapshot(broker_unknown=False), _incident_snapshot(broker_unknown=True)],
    )

    assert [result["sent"] for result in results] == [True, True]
