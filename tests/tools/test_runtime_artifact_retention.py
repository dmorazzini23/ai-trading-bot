from __future__ import annotations

import json
from pathlib import Path

from ai_trading.tools import runtime_artifact_retention
from ai_trading.tools.runtime_artifact_retention import (
    RetentionRule,
    evaluate_runtime_artifact_retention,
    main,
)


def _write_jsonl(path: Path, rows: int) -> None:
    path.write_text(
        "".join(json.dumps({"row": idx}) + "\n" for idx in range(rows)),
        encoding="utf-8",
    )


def test_retention_plan_does_not_mutate_large_file(tmp_path):
    path = tmp_path / "events.jsonl"
    _write_jsonl(path, 20)
    before = path.read_text(encoding="utf-8")

    report = evaluate_runtime_artifact_retention(
        runtime_dir=tmp_path,
        apply=False,
        rules=(RetentionRule("events.jsonl", 20, 5),),
    )

    assert report["status"] == "planned"
    assert report["actions"][0]["status"] == "would_compact"
    assert path.read_text(encoding="utf-8") == before


def test_retention_apply_keeps_tail_and_writes_backup(tmp_path):
    path = tmp_path / "events.jsonl"
    _write_jsonl(path, 20)

    report = evaluate_runtime_artifact_retention(
        runtime_dir=tmp_path,
        apply=True,
        rules=(RetentionRule("events.jsonl", 20, 5),),
    )

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert report["status"] == "applied"
    assert report["actions"][0]["status"] == "compacted"
    assert [row["row"] for row in rows] == [15, 16, 17, 18, 19]
    assert list(tmp_path.glob("events.jsonl.bak.*.gz"))


def test_retention_cli_writes_report(tmp_path):
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    output = tmp_path / "retention.json"

    rc = main(["--runtime-dir", str(runtime_dir), "--output-json", str(output)])

    assert rc == 0
    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8"))["artifact_type"] == "runtime_artifact_retention"


def test_retention_cli_prefers_canonical_runtime(monkeypatch, tmp_path):
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    repo_runtime = tmp_path / "repo" / "runtime"
    repo_runtime.mkdir(parents=True)
    output = tmp_path / "retention.json"
    monkeypatch.chdir(tmp_path / "repo")
    monkeypatch.setattr(runtime_artifact_retention, "_CANONICAL_RUNTIME_DIR", canonical)

    rc = runtime_artifact_retention.main(["--output-json", str(output)])

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["runtime_dir"] == str(canonical.resolve())
