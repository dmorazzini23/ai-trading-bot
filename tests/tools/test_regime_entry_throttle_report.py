from __future__ import annotations

from datetime import UTC, datetime

from ai_trading.tools.regime_entry_throttle_report import build_report


def test_regime_entry_throttle_report_uses_live_cost_evidence() -> None:
    report = build_report(
        live_cost_model={
            "generated_at": "2026-05-08T12:00:00Z",
            "by_symbol_side_session": [
                {
                    "symbol": "AMZN",
                    "side": "buy",
                    "session_regime": "midday",
                    "sample_count": 40,
                    "p90_spread_bps": 4.0,
                }
            ],
        },
        health={"data_provider": {"status": "healthy"}},
        report_date="2026-05-08",
        enforce=False,
        live_canary=False,
        now=datetime(2026, 5, 8, 12, 0, tzinfo=UTC),
    )

    assert report["artifact_type"] == "regime_entry_throttle_report"
    assert report["mode"] == "report_only"
    assert report["actions"] == {"observe": 1}
    assert report["sample_sufficiency"]["insufficient"] == 0


def test_regime_entry_throttle_report_can_fail_closed_when_enforced() -> None:
    report = build_report(
        live_cost_model={"generated_at": "2026-05-08T12:00:00Z"},
        health={"data_provider": {"status": "degraded"}},
        report_date="2026-05-08",
        enforce=True,
        live_canary=True,
        now=datetime(2026, 5, 8, 12, 0, tzinfo=UTC),
    )

    assert report["actions"] == {"block_new_entries": 1}
    assert report["latest"]["qty_scale"] == 0.0
