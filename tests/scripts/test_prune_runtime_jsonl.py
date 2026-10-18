from __future__ import annotations

import gzip
import json
import os
import re
import subprocess
import tarfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ai_trading.tools.runtime_artifact_retention import _DEFAULT_RULES

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "prune_runtime_jsonl.sh"
DAY = 86_400


def _env(data_dir: Path, **overrides: str) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("AI_TRADING_")}
    env.update(
        {
            "AI_TRADING_DATA_DIR": str(data_dir),
            "AI_TRADING_RUNTIME_DIR": str(data_dir / "runtime"),
            "AI_TRADING_RUNTIME_PRUNE_SETTLE_SECONDS": "0",
        }
    )
    env.update(overrides)
    return env


def _run(data_dir: Path, *args: str, **overrides: str) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        ["bash", str(SCRIPT), *args],
        env=_env(data_dir, **overrides),
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    return proc


def _real_now() -> float:
    # The suite may freeze Python clocks; the script under test uses the real one.
    proc = subprocess.run(["date", "-u", "+%s"], capture_output=True, text=True, check=True)
    return float(proc.stdout.strip())


def _age(path: Path, days: float) -> None:
    stamp = _real_now() - days * DAY
    for entry in sorted(path.rglob("*"), reverse=True):
        os.utime(entry, (stamp, stamp), follow_symlinks=False)
    os.utime(path, (stamp, stamp))


def _run_name(days_ago: float, suffix: str = "daily") -> str:
    stamp = datetime.fromtimestamp(_real_now(), tz=UTC) - timedelta(days=days_ago)
    return stamp.strftime("%Y%m%dT%H%M%SZ") + f"_{suffix}"


def _make_run(root: Path, cadence: str, name: str, *, days_old: float) -> Path:
    run = root / cadence / name
    (run / "nested").mkdir(parents=True)
    (run / "trading_day_report.json").write_text(json.dumps({"run": name}), encoding="utf-8")
    (run / "nested" / "rows.jsonl").write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
    _age(run, days_old)
    return run


def _write_jsonl(path: Path, rows: int) -> bytes:
    payload = "".join(json.dumps({"row": idx, "pad": "x" * 30}) + "\n" for idx in range(rows))
    path.write_text(payload, encoding="utf-8")
    return payload.encode("utf-8")


def _tree_state(root: Path) -> list[tuple[str, int, int]]:
    return sorted(
        (str(entry.relative_to(root)), entry.lstat().st_size if entry.is_file() else -1, entry.lstat().st_mtime_ns)
        for entry in root.rglob("*")
    )


def test_dry_run_prints_plan_and_changes_nothing(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    _write_jsonl(runtime / "decision_records.jsonl", 200)
    _write_jsonl(runtime / "gate_effectiveness.jsonl", 5)
    old_run = _run_name(45)
    _make_run(runtime / "research_reports", "daily", old_run, days_old=45)
    before = _tree_state(tmp_path)

    proc = _run(tmp_path, "--dry-run", AI_TRADING_DECISION_RECORDS_MAX_BYTES="4000")

    assert "jsonl decision_records.jsonl: size=" in proc.stdout
    assert "action=rotate" in proc.stdout
    assert "jsonl gate_effectiveness.jsonl: size=" in proc.stdout and "reason=within_limit" in proc.stdout
    assert f"research_reports daily/{old_run}:" in proc.stdout
    assert "action=archive" in proc.stdout
    assert "mode=dry-run" in proc.stdout
    assert _tree_state(tmp_path) == before


def test_apply_rotates_into_compressed_archive_and_never_deletes(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    original = _write_jsonl(runtime / "decision_records.jsonl", 200)
    within = _write_jsonl(runtime / "gate_effectiveness.jsonl", 5)
    archive_dir = runtime / "recovery_backups" / "archive" / "jsonl" / "decision_records"
    archive_dir.mkdir(parents=True)
    old_archive = archive_dir / "decision_records.jsonl.20200101T000000Z.gz"
    old_archive.write_bytes(gzip.compress(b'{"old": true}\n'))
    legacy = runtime / "decision_records.jsonl.bak.20200101T000000Z.gz"
    legacy.write_bytes(gzip.compress(b'{"legacy": true}\n'))
    for path in (old_archive, legacy):
        stamp = _real_now() - 60 * DAY
        os.utime(path, (stamp, stamp))

    proc = _run(
        tmp_path,
        AI_TRADING_DECISION_RECORDS_MAX_BYTES="4000",
        AI_TRADING_DECISION_RECORDS_KEEP_LINES="1000",
    )

    live = (runtime / "decision_records.jsonl").read_bytes()
    assert 0 < len(live) <= 4000
    assert original.endswith(live)
    assert all(json.loads(line)["row"] >= 0 for line in live.decode().splitlines())
    new_archives = sorted(set(archive_dir.glob("*.gz")) - {old_archive})
    assert len(new_archives) == 1
    assert gzip.decompress(new_archives[0].read_bytes()) == original
    assert old_archive.exists() and legacy.exists()
    assert (runtime / "gate_effectiveness.jsonl").read_bytes() == within
    assert not list(runtime.glob("*.tmp.*"))
    assert "jsonl decision_records.jsonl: rotated" in proc.stdout


def test_apply_respects_keep_lines_cap(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    _write_jsonl(runtime / "execution_quality_events.jsonl", 300)

    _run(
        tmp_path,
        AI_TRADING_EXECUTION_QUALITY_EVENTS_MAX_BYTES="8000",
        AI_TRADING_EXECUTION_QUALITY_EVENTS_KEEP_LINES="10",
    )

    rows = [json.loads(line) for line in (runtime / "execution_quality_events.jsonl").read_text().splitlines()]
    assert [row["row"] for row in rows] == list(range(290, 300))


def test_apply_archives_only_unreferenced_research_runs_older_than_cutoff(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    reports = runtime / "research_reports"
    old_run = _run_name(45)
    referenced_run = _run_name(44, "weekend-saturday")
    recent_run = _run_name(5)
    touched_run = _run_name(43)
    old_dir = _make_run(reports, "daily", old_run, days_old=45)
    expected_files = {
        str(path.relative_to(old_dir.parent)): path.read_bytes() for path in old_dir.rglob("*") if path.is_file()
    }
    _make_run(reports, "weekend", referenced_run, days_old=44)
    _make_run(reports, "daily", recent_run, days_old=5)
    touched_dir = _make_run(reports, "daily", touched_run, days_old=43)
    (touched_dir / "late.json").write_text("{}", encoding="utf-8")
    latest = reports / "latest"
    latest.mkdir(parents=True)
    (latest / "daily_research_latest.json").write_text("{}", encoding="utf-8")
    _age(latest, 90)
    cache = reports / "huggingface" / "cache"
    cache.mkdir(parents=True)
    (cache / "weights.bin").write_bytes(b"0" * 16)
    _age(reports / "huggingface", 90)
    models = tmp_path / "models"
    models.mkdir()
    (models / "registry_index.json").write_text(
        json.dumps({"m": {"artifact_path": f"runtime/research_reports/weekend/{referenced_run}/model.joblib"}}),
        encoding="utf-8",
    )

    proc = _run(tmp_path)

    archive = runtime / "recovery_backups" / "archive" / "research_reports" / "daily" / f"{old_run}.tar.gz"
    assert archive.is_file()
    assert not old_dir.exists()
    with tarfile.open(archive, "r:gz") as bundle:
        extracted = {
            member.name: bundle.extractfile(member).read()  # type: ignore[union-attr]
            for member in bundle.getmembers()
            if member.isfile()
        }
    assert extracted == expected_files
    assert (reports / "weekend" / referenced_run).is_dir()
    assert (reports / "daily" / recent_run).is_dir()
    assert touched_dir.is_dir()
    assert (latest / "daily_research_latest.json").is_file()
    assert (cache / "weights.bin").is_file()
    assert "archive_candidates=1" in proc.stdout
    assert "kept_referenced=1" in proc.stdout


def test_research_archive_honours_per_run_cap_and_disable_switch(tmp_path: Path) -> None:
    reports = tmp_path / "runtime" / "research_reports"
    runs = [_run_name(40 + offset) for offset in range(3)]
    for name in runs:
        _make_run(reports, "daily", name, days_old=45)

    disabled = _run(tmp_path, AI_TRADING_RESEARCH_REPORT_ARCHIVE_ENABLED="0")
    assert "research_reports: action=none reason=disabled" in disabled.stdout
    assert all((reports / "daily" / name).is_dir() for name in runs)

    capped = _run(tmp_path, AI_TRADING_RESEARCH_REPORT_ARCHIVE_MAX_DIRS="1")
    remaining = [name for name in runs if (reports / "daily" / name).is_dir()]
    assert len(remaining) == 2
    assert "archived=1 deferred_to_next_run=2" in capped.stdout


def test_rejects_unknown_arguments(tmp_path: Path) -> None:
    proc = subprocess.run(
        ["bash", str(SCRIPT), "--force"],
        env=_env(tmp_path),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert proc.returncode == 2
    assert "unknown argument" in proc.stderr


def test_script_caps_match_python_retention_planner() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    pattern = re.compile(
        r'rotate_jsonl_file \\\n\s+"\$\{RUNTIME_DIR\}/(?P<name>[a-z_]+\.jsonl)" \\\n'
        r'\s+"\$\{[A-Z_]+:-(?P<max>\d+)\}" \\\n\s+"\$\{[A-Z_]+:-(?P<keep>\d+)\}"'
    )
    script_rules = {m["name"]: (int(m["max"]), int(m["keep"])) for m in pattern.finditer(text)}
    planner_rules = {rule.filename: (rule.max_bytes, rule.keep_lines) for rule in _DEFAULT_RULES}
    assert len(script_rules) == 12
    assert script_rules == planner_rules
