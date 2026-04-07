from __future__ import annotations

import os
from pathlib import Path

from ai_trading.runtime import artifacts as runtime_artifacts
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path


def test_resolve_runtime_artifact_path_prefers_data_dir(
    monkeypatch,
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data-root"
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(data_dir))
    monkeypatch.delenv("STATE_DIRECTORY", raising=False)

    resolved = resolve_runtime_artifact_path(
        "runtime/research_reports/report.json",
        default_relative="runtime/research_reports/report.json",
    )

    assert resolved == (data_dir / "runtime/research_reports/report.json").resolve()


def test_resolve_runtime_artifact_path_keeps_absolute(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("AI_TRADING_DATA_DIR", raising=False)
    monkeypatch.delenv("STATE_DIRECTORY", raising=False)
    target = (tmp_path / "absolute.json").resolve()

    resolved = resolve_runtime_artifact_path(
        str(target),
        default_relative="runtime/ignored.json",
    )

    assert resolved == target


def test_resolve_runtime_artifact_path_for_write_uses_data_dir(
    monkeypatch,
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "runtime-root"
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(data_dir))
    monkeypatch.delenv("STATE_DIRECTORY", raising=False)

    resolved = resolve_runtime_artifact_path(
        "runtime/order_events.jsonl",
        default_relative="runtime/order_events.jsonl",
        for_write=True,
    )

    assert resolved == (data_dir / "runtime/order_events.jsonl").resolve()


def test_resolve_runtime_artifact_path_prefers_newest_existing_candidate(
    monkeypatch,
    tmp_path: Path,
) -> None:
    primary_root = tmp_path / "primary-root"
    secondary_root = tmp_path / "secondary-root"
    monkeypatch.delenv("AI_TRADING_DATA_DIR", raising=False)
    monkeypatch.setattr(
        runtime_artifacts,
        "_iter_runtime_roots",
        lambda: [primary_root, secondary_root],
    )

    relative_path = "runtime/runtime_performance_report_latest.json"
    primary_target = (primary_root / relative_path).resolve()
    secondary_target = (secondary_root / relative_path).resolve()
    primary_target.parent.mkdir(parents=True, exist_ok=True)
    secondary_target.parent.mkdir(parents=True, exist_ok=True)
    primary_target.write_text("{}", encoding="utf-8")
    secondary_target.write_text("{}", encoding="utf-8")

    # Force deterministic "newer" ordering.
    os.utime(primary_target, (1_700_000_000, 1_700_000_000))
    os.utime(secondary_target, (2_100_000_000, 2_100_000_000))

    resolved = resolve_runtime_artifact_path(
        relative_path,
        default_relative=relative_path,
    )

    assert resolved == secondary_target
