from __future__ import annotations

import json

from ai_trading.tools import runtime_gonogo_status as status_tool


def test_format_status_line_includes_core_fields() -> None:
    payload = {
        "gate_passed": False,
        "failed_checks": ["trade_used_days", "profit_factor"],
        "thresholds": {
            "lookback_days": 5,
            "min_used_days": 3,
            "trade_fill_source": "live",
        },
        "observed": {
            "trade_used_days": 2,
            "gate_used_days": 3,
            "closed_trades": 197,
            "profit_factor": 0.57,
            "win_rate": 0.47,
            "net_pnl": -281.73,
            "acceptance_rate": 0.032,
        },
    }

    line = status_tool.format_status_line(payload)

    assert "state=FAIL" in line
    assert "failed=trade_used_days,profit_factor" in line
    assert "lookback_days=5" in line
    assert "min_used_days=3" in line
    assert "fill_source=live" in line
    assert "trade_used_days=2" in line
    assert "gate_used_days=3" in line


def test_resolve_thresholds_prefers_execution_trade_fill_source(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "AI_TRADING_EXECUTION_RUNTIME_GONOGO_TRADE_FILL_SOURCE",
        "live",
    )
    monkeypatch.setenv(
        "AI_TRADING_RUNTIME_GONOGO_TRADE_FILL_SOURCE",
        "all",
    )

    thresholds = status_tool._resolve_thresholds()

    assert thresholds["trade_fill_source"] == "live"


def test_main_returns_fail_code_with_one_line(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        status_tool,
        "build_runtime_gonogo_status",
        lambda: {
            "gate_passed": False,
            "failed_checks": ["trade_used_days"],
            "thresholds": {"lookback_days": 5, "min_used_days": 3},
            "observed": {"trade_used_days": 2, "gate_used_days": 3},
        },
    )

    exit_code = status_tool.main([])
    line = capsys.readouterr().out.strip()

    assert exit_code == 2
    assert line.startswith("RUNTIME_GONOGO_STATUS state=FAIL ")
    assert "failed=trade_used_days" in line


def test_main_json_pass(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        status_tool,
        "build_runtime_gonogo_status",
        lambda: {
            "gate_passed": True,
            "failed_checks": [],
            "checks": {"trade_used_days": True},
            "thresholds": {"lookback_days": 5, "min_used_days": 3},
            "observed": {"trade_used_days": 3, "gate_used_days": 3},
            "paths": {"trade_history": "/tmp/trade_history.parquet"},
        },
    )

    exit_code = status_tool.main(["--json"])
    output = capsys.readouterr().out.strip()
    payload = json.loads(output)

    assert exit_code == 0
    assert payload["gate_passed"] is True
    assert payload["observed"]["trade_used_days"] == 3
