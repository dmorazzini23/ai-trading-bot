from __future__ import annotations

import json
import importlib.util
from io import BytesIO
from urllib.error import HTTPError
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from datetime import UTC, datetime

import pytest


def _load_recap_module() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts" / "openclaw_market_close_recap.py"
    spec = importlib.util.spec_from_file_location("openclaw_market_close_recap_test_module", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recap = _load_recap_module()


def test_recap_explains_diagnostics_without_hiding_readiness_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(recap, 'RUNTIME_DIR', tmp_path)
    monkeypatch.setattr(recap, '_service_summary', lambda: 'ai-trading.service active')
    monkeypatch.setattr(recap, '_journal_summary', lambda day: 'No errors')
    monkeypatch.setattr(recap, '_health_summary', lambda: ('degraded', {
        'ok': False, 'status': 'degraded', 'readiness_failures': ['required_model_stale'],
        'broker': {'connected': True, 'positions_count': 0, 'open_orders_count': 0},
        'operating_mode': {'summary': 'Paper diagnostics enabled; qualified model unavailable.'},
    }))
    text = recap.build_recap()
    assert 'required_model_stale' in text
    assert 'Paper diagnostics enabled; qualified model unavailable.' in text
    assert 'Healthy close' not in text


def test_build_recap_reports_fresh_artifacts_without_system_checks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime = tmp_path / "runtime"
    reports = runtime / "reports"
    reports.mkdir(parents=True)
    (runtime / "fill_events.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "fill_recorded",
                        "ts": "2026-06-29T15:00:00+00:00",
                        "symbol": "QQQ",
                        "side": "buy",
                        "fill_qty": 1,
                        "realized_net_edge_bps": 2.5,
                    }
                ),
                json.dumps(
                    {
                        "event": "fill_recorded",
                        "ts": "2026-06-29T15:05:00+00:00",
                        "symbol": "QQQ",
                        "side": "sell",
                        "fill_qty": 1,
                        "realized_net_edge_bps": 3.5,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    (reports / "trading_day_latest.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-06-29T20:37:00Z",
                "report_date": "2026-06-29",
                "openclaw_summary": {
                    "summary": "trading_day desired=2 submitted=2 rejected=0 controlled_skips=0 fills=2",
                    "details": {
                        "desired": 2,
                        "submitted": 2,
                        "fills": 2,
                        "rejected": 0,
                        "symbols_with_activity": ["QQQ"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AI_TRADING_RECAP_RUNTIME_DIR", str(runtime))
    monkeypatch.setenv("AI_TRADING_RECAP_DATE", "2026-06-29")
    monkeypatch.setenv("AI_TRADING_RECAP_SKIP_SYSTEM", "1")
    monkeypatch.setattr(recap, "RUNTIME_DIR", runtime)

    text = recap.build_recap()

    assert "**Close verdict**" in text
    assert "**Trading snapshot**" in text
    assert "2 fills" in text
    assert "realized edge sum 6.00 bps" in text
    assert "trading_day desired=2" in text


def test_build_recap_marks_stale_trading_day_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime = tmp_path / "runtime"
    reports = runtime / "reports"
    reports.mkdir(parents=True)
    (runtime / "fill_events.jsonl").write_text("", encoding="utf-8")
    (reports / "trading_day_latest.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-06-28T20:37:00Z",
                "report_date": "2026-06-28",
                "openclaw_summary": {"summary": "old report"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AI_TRADING_RECAP_RUNTIME_DIR", str(runtime))
    monkeypatch.setenv("AI_TRADING_RECAP_DATE", "2026-06-29")
    monkeypatch.setenv("AI_TRADING_RECAP_SKIP_SYSTEM", "1")
    monkeypatch.setattr(recap, "RUNTIME_DIR", runtime)

    text = recap.build_recap()

    assert "trading-day report stale" in text
    assert "report_date=2026-06-28" in text
    assert "trading-day report pending for 2026-06-29" in text
    assert "old report" not in text


def test_health_summary_parses_degraded_http_error_payload(monkeypatch) -> None:
    payload = {
        "ok": False,
        "status": "healthy",
        "reason": "market_closed",
        "broker": {
            "connected": True,
            "open_orders_count": 0,
            "positions_count": 0,
        },
        "provider_state": {"active": "alpaca", "status": "warming_up"},
        "readiness_failures": ["replay_live_parity_gate_failed"],
    }

    def _urlopen(*_args, **_kwargs):
        raise HTTPError(
            recap.HEALTH_URL,
            503,
            "SERVICE UNAVAILABLE",
            hdrs=None,
            fp=BytesIO(json.dumps(payload).encode("utf-8")),
        )

    monkeypatch.delenv("AI_TRADING_RECAP_SKIP_SYSTEM", raising=False)
    monkeypatch.setattr(recap, "urlopen", _urlopen)

    summary, observed = recap._health_summary()

    assert observed == payload
    assert "health ok=False status=healthy reason=market_closed" in summary
    assert "readiness_failures=replay_live_parity_gate_failed" in summary
    assert "health unavailable" not in summary
    monkeypatch.setattr(recap, "_health_summary", lambda: (summary, payload))
    monkeypatch.setattr(recap, "_service_summary", lambda: "service active")
    monkeypatch.setattr(recap, "_fill_summary", lambda _day: {"available": False})
    monkeypatch.setattr(
        recap,
        "_trading_day_summary",
        lambda _day: ("trading-day report pending", []),
    )
    monkeypatch.setattr(recap, "_journal_summary", lambda _day: "journal clean")
    monkeypatch.setattr(recap, "_operator_issues", lambda: [])

    text = recap.build_recap()
    assert "readiness failures: replay_live_parity_gate_failed" in text


def test_journal_summary_ignores_broker_last_error_field(monkeypatch) -> None:
    normal = json.dumps(
        {
            "level": "INFO",
            "msg": "CYCLE_MARKET_SNAPSHOT",
            "broker_last_error": None,
        }
    )
    failure = json.dumps(
        {
            "level": "ERROR",
            "msg": "after-hours training failed",
        }
    )
    monkeypatch.delenv("AI_TRADING_RECAP_SKIP_SYSTEM", raising=False)
    monkeypatch.setattr(
        recap,
        "_run_command",
        lambda *_args, **_kwargs: (0, f"{normal}\n{failure}"),
    )

    summary = recap._journal_summary("2026-07-13")

    assert summary.startswith("1 relevant events in the bounded journal sample")
    assert "after-hours training failed" in summary
    assert "broker_last_error" not in summary


def test_reset_recap_uses_current_paper_review_not_retired_report(tmp_path, monkeypatch):
    latest = tmp_path / "research_reports" / "latest"
    latest.mkdir(parents=True)
    (latest / "daily_research_automation_latest.json").write_text(json.dumps({
        "generated_at": "2026-09-17T20:38:00Z", "status": "complete",
        "steps": [{"name": "research_reset_scorecard"}],
    }))
    paper = latest / "paper_evidence_review_latest.json"
    paper.write_text(json.dumps({"session_date": "2026-09-17",
        "generated_at": "2026-09-17T20:37:00Z", "status": "evidence_pending",
        "completion_gaps": ["net_cost_validation_unavailable"]}))
    monkeypatch.setattr(recap, "RUNTIME_DIR", tmp_path)
    monkeypatch.setenv("AI_TRADING_RECAP_DATE", "2026-09-17")
    text, issues = recap._trading_day_summary("2026-09-17")
    assert "paper-session review evidence_pending" in text
    assert "paused under research reset" in text
    assert issues == ["Paper evidence: net_cost_validation_unavailable"]
    (tmp_path / "operator_control_plane_latest.json").write_text(json.dumps({
        "status": "complete", "generated_at": "2026-09-07T21:00:00Z"}))
    assert recap._operator_issues() == []
    text, issues = recap._trading_day_summary("2026-09-18")
    assert "pending" in text and issues == ["Reset evidence reports missing or stale"]


def test_incident_checked_at_and_stale_operator_are_distinguished(tmp_path, monkeypatch):
    monkeypatch.setattr(recap, "RUNTIME_DIR", tmp_path)
    monkeypatch.setenv("AI_TRADING_RECAP_DATE", "2026-09-17")
    (tmp_path / "openclaw_incident_state.json").write_text(json.dumps({
        "status": "clear", "checked_at": "2026-09-17T21:00:00Z"}))
    (tmp_path / "operator_control_plane_latest.json").write_text(json.dumps({
        "status": "complete", "generated_at": "2026-09-07T21:00:00Z"}))
    issues = recap._operator_issues()
    assert "historical/unverified" in issues[0]
    assert issues[1] == "OpenClaw incident state: clear at 2026-09-17T21:00:00Z"


def test_fill_last_timestamp_only_uses_matching_session_fills(tmp_path, monkeypatch):
    rows = [
        {"event": "fill_recorded", "ts": "2026-09-17T14:00:00Z", "qty": 1},
        {"event": "other", "ts": "2026-09-17T19:00:00Z"},
        {"event": "fill_recorded", "ts": "2026-09-18T00:30:00Z", "qty": 1},
        {"event": "fill_recorded", "ts": "2026-09-18T14:00:00Z", "qty": 1},
    ]
    (tmp_path / "fill_events.jsonl").write_text("\n".join(map(json.dumps, rows)))
    monkeypatch.setattr(recap, "RUNTIME_DIR", tmp_path)
    result = recap._fill_summary("2026-09-17")
    assert result["fills"] == 2
    assert result["last_ts"] == "2026-09-18T00:30:00Z"
    assert recap._fill_summary("2026-09-19")["last_ts"] is None


def test_journal_filters_whole_new_york_day_before_limiting(monkeypatch):
    observed = []
    monkeypatch.delenv("AI_TRADING_RECAP_SKIP_SYSTEM", raising=False)
    monkeypatch.delenv("AI_TRADING_RECAP_JOURNAL_SINCE", raising=False)
    monkeypatch.setattr(recap, "_run_command", lambda args, **kw: (observed.append(args) or (0, "")))
    assert "does not establish absence of fills" in recap._journal_summary("2026-09-17")
    args = observed[0]
    assert args[args.index("--since") + 1] == "2026-09-17 04:00:00 UTC"
    assert args[args.index("--until") + 1] == "2026-09-18 04:00:00 UTC"
    assert "--grep" in args


def test_session_date_does_not_roll_at_utc_midnight(monkeypatch):
    monkeypatch.delenv("AI_TRADING_RECAP_DATE", raising=False)
    monkeypatch.setattr(recap, "datetime", SimpleNamespace(
        now=lambda tz: datetime(2026, 9, 18, 0, 10, tzinfo=UTC).astimezone(tz)))
    assert recap._utc_today() == "2026-09-17"


@pytest.mark.parametrize("connected,failures,active,healthy", [
    (True, [], True, True), (False, [], True, False),
    (True, ["required_model_stale"], True, False), (True, [], False, False),
])
def test_healthy_close_requires_connection_readiness_and_active_service(
    monkeypatch, connected, failures, active, healthy,
):
    monkeypatch.setattr(recap, "_health_summary", lambda: ("health", {
        "ok": True, "readiness_failures": failures,
        "broker": {"connected": connected, "positions_count": 0, "open_orders_count": 0},
    }))
    monkeypatch.setattr(recap, "_service_summary", lambda: "ai-trading.service active" if active else "inactive")
    monkeypatch.setattr(recap, "_fill_summary", lambda day: {"available": False})
    monkeypatch.setattr(recap, "_trading_day_summary", lambda day: ("pending", []))
    monkeypatch.setattr(recap, "_journal_summary", lambda day: "journal")
    monkeypatch.setattr(recap, "_operator_issues", lambda: [])
    assert ("Healthy close:" in recap.build_recap()) is healthy
