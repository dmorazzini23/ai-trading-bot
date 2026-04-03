from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from tools import mcp_runtime_data_server as runtime_srv


def test_extract_json_objects_filters_non_json_lines() -> None:
    lines = [
        "plain text",
        '{"ok": true, "a": 1}',
        "still not json",
        '{"gate_passed": false}',
    ]
    objs = runtime_srv._extract_json_objects(lines)
    assert len(objs) == 2
    assert objs[-1]["gate_passed"] is False


def test_extract_json_objects_parses_multiline_json_with_log_prefix() -> None:
    payload = (
        '{"ts":"2026-04-03T01:00:00Z","level":"INFO","msg":"startup"}\n'
        "{\n"
        '  "go_no_go": {"gate_passed": true},\n'
        '  "top_gates": [{"gate":"runtime_gonogo_gate","count":3}]\n'
        "}\n"
    )
    objs = runtime_srv._extract_json_objects(payload)
    assert len(objs) == 2
    assert objs[-1]["go_no_go"]["gate_passed"] is True


def test_run_module_json_handles_pretty_report_after_structured_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = (
        '{"ts":"2026-04-03T01:00:00Z","level":"INFO","msg":"logger-ready"}\n'
        "{\n"
        '  "go_no_go": {"gate_passed": true},\n'
        '  "execution_vs_alpha": {"edge_realism_gap_ratio": 0.25}\n'
        "}\n"
    )

    def _fake_run(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(runtime_srv.subprocess, "run", _fake_run)
    payload = runtime_srv._run_module_json("ai_trading.tools.runtime_performance_report", ["--json"])
    assert payload["go_no_go"]["gate_passed"] is True
    assert payload["execution_vs_alpha"]["edge_realism_gap_ratio"] == pytest.approx(0.25)


def test_safe_runtime_path_blocks_escape(tmp_path: Path) -> None:
    root = tmp_path / "runtime"
    root.mkdir()
    with pytest.raises(ValueError):
        runtime_srv._safe_runtime_path(root, "../secrets.json")


def test_trade_history_summary_supports_pickle(tmp_path: Path) -> None:
    root = tmp_path / "runtime"
    root.mkdir()
    path = root / "trade_history.pkl"
    frame = pd.DataFrame(
        {
            "fill_source": ["live", "live", "unknown"],
            "pnl": [1.0, -0.5, 0.2],
        }
    )
    frame.to_pickle(path)

    summary = runtime_srv.tool_trade_history_summary(
        {"runtime_root": str(root), "path": "trade_history.pkl"}
    )
    assert summary["exists"] is True
    assert summary["rows"] == 3
    assert summary["fill_source_counts"]["live"] == 2
    assert summary["pnl_sum"] == pytest.approx(0.7)
