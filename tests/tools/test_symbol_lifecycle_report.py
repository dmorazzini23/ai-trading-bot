from __future__ import annotations

from ai_trading.tools import symbol_lifecycle_report as report_tool


def test_symbol_lifecycle_requires_manual_approval_for_authority_increase() -> None:
    payload = report_tool.build_symbol_lifecycle_report(
        report_date="2026-05-05",
        symbols=["AMZN"],
        shadow_report={
            "markout_summary": {
                "best_symbols": [
                    {
                        "symbol": "AMZN",
                        "samples": 30,
                        "mean_net_markout_bps": 4.0,
                        "positive_rate": 0.60,
                    }
                ]
            }
        },
        shadow_symbols=["AMZN"],
        min_samples=25,
    )

    row = payload["symbols"][0]
    assert row["recommendation"] == "consider_canary"
    assert row["recommended_mode"] == "canary"
    assert row["manual_approval_required"] is True
    assert payload["runtime_symbol_gating_changed"] is False


def test_symbol_lifecycle_restricts_negative_allowed_symbol() -> None:
    payload = report_tool.build_symbol_lifecycle_report(
        report_date="2026-05-05",
        symbols=["AAPL"],
        replay_report={
            "replay_symbol_summary": {
                "AAPL": {"sample_count": 40, "net_edge_bps": -12.0, "win_rate": 0.35}
            }
        },
        allow_symbols=["AAPL"],
        min_samples=25,
    )

    row = payload["symbols"][0]
    assert row["recommendation"] == "move_to_shadow_only"
    assert row["recommended_mode"] == "shadow_only"
    assert row["manual_approval_required"] is False
