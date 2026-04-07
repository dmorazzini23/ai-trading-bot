from __future__ import annotations

from pathlib import Path

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
