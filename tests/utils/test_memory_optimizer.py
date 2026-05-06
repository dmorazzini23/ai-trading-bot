from __future__ import annotations

import json

from ai_trading.utils.memory_optimizer import report_memory_use


def test_report_memory_use_writes_bounded_sample(tmp_path):
    sample_path = tmp_path / "memory_samples.jsonl"

    snapshot = report_memory_use(
        cycle_index=7,
        closed=False,
        interval_s=85.0,
        write_sample=True,
        sample_path=sample_path,
        max_bytes=10_000,
    )

    assert snapshot["cycle_index"] == 7
    assert snapshot["closed"] is False
    assert snapshot["interval_s"] == 85.0
    assert snapshot["pid"] > 0
    assert "rss_mb" in snapshot
    assert snapshot["level"] in {"normal", "warning", "critical", "unknown"}
    assert snapshot["sample_path"] == str(sample_path)

    rows = [json.loads(line) for line in sample_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["cycle_index"] == 7
    assert rows[0]["gc_counts"]


def test_report_memory_use_can_collect_objects_without_writing(tmp_path):
    sample_path = tmp_path / "memory_samples.jsonl"

    snapshot = report_memory_use(sample_path=sample_path, collect=True)

    assert snapshot["objects_collected"] >= 0
    assert "sample_path" not in snapshot
    assert not sample_path.exists()
