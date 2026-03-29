from __future__ import annotations

from pathlib import Path

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

