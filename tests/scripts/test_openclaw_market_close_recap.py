from __future__ import annotations

import json
import importlib.util
from io import BytesIO
from urllib.error import HTTPError
from pathlib import Path
from types import ModuleType


def _load_recap_module() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts" / "openclaw_market_close_recap.py"
    spec = importlib.util.spec_from_file_location("openclaw_market_close_recap_test_module", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recap = _load_recap_module()


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

    assert summary.startswith("1 matching journal lines")
    assert "after-hours training failed" in summary
    assert "broker_last_error" not in summary
