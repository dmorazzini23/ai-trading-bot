from __future__ import annotations

import json
import importlib.util
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
