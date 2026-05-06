from __future__ import annotations

import json

from ai_trading.tools.memory_hotspot_audit import build_memory_hotspot_audit


def test_memory_hotspot_audit_reports_runtime_files_samples_and_code_patterns(tmp_path):
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    sample_path = runtime_dir / "memory_samples.jsonl"
    sample_path.write_text(
        "\n".join(
            [
                json.dumps({"ts": "2026-05-05T00:00:00Z", "rss_mb": 100.0, "level": "normal"}),
                json.dumps({"ts": "2026-05-05T00:01:00Z", "rss_mb": 125.5, "level": "normal"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (runtime_dir / "large_runtime.jsonl").write_text("x" * 2048, encoding="utf-8")
    repo_root = tmp_path / "repo"
    code_dir = repo_root / "ai_trading"
    code_dir.mkdir(parents=True)
    (code_dir / "reader.py").write_text(
        "from pathlib import Path\npayload = Path('x.jsonl').read_text()\n",
        encoding="utf-8",
    )

    report = build_memory_hotspot_audit(
        runtime_dir=runtime_dir,
        repo_root=repo_root,
        sample_path=sample_path,
        service="definitely-not-a-real-service.service",
        max_files=5,
        max_code_findings=5,
        sample_lines=10,
    )

    assert report["artifact_type"] == "memory_hotspot_audit"
    assert report["recent_memory_samples"]["rows"] == 2
    assert report["recent_memory_samples"]["rss_delta_mb"] == 25.5
    assert report["runtime_artifacts"]["largest_file"]["path"].endswith("large_runtime.jsonl")
    assert report["code_hotspots"][0]["path"].endswith("reader.py")
    assert "whole_file_reader_patterns_present" in report["observations"]
