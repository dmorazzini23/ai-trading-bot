from __future__ import annotations

import subprocess

import pytest

import tools.harvest_import_errors as harvest


def test_build_report_returns_collection_status(monkeypatch, tmp_path):
    monkeypatch.setattr(harvest, "compute_env_summary_line", lambda: "env")
    monkeypatch.setattr(harvest, "assert_expected_combo", lambda _line: None)
    monkeypatch.setattr(
        harvest.Path, "resolve", lambda self: tmp_path / "tools" / "harvest_import_errors.py"
    )

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=[], returncode=3, stdout="collection failed", stderr=""
        )

    monkeypatch.setattr(harvest.subprocess, "run", fake_run)

    _report, raw_text, _env_line, returncode = harvest.build_report()

    assert "collection failed" in raw_text
    assert returncode == 3
    assert "Pytest collection return code: 3" in _report


def test_build_report_times_out_collection(monkeypatch, tmp_path):
    monkeypatch.setattr(harvest, "compute_env_summary_line", lambda: "env")
    monkeypatch.setattr(harvest, "assert_expected_combo", lambda _line: None)
    monkeypatch.setattr(harvest, "_COLLECT_TIMEOUT_S", 7)
    monkeypatch.setattr(
        harvest.Path, "resolve", lambda self: tmp_path / "tools" / "harvest_import_errors.py"
    )

    def fake_run(*_args, **kwargs):
        assert kwargs["timeout"] == 7
        raise subprocess.TimeoutExpired(
            cmd=kwargs.get("args", "pytest"),
            timeout=7,
            output="partial stdout",
            stderr="partial stderr",
        )

    monkeypatch.setattr(harvest.subprocess, "run", fake_run)

    _report, raw_text, _env_line, returncode = harvest.build_report()

    assert returncode == 124
    assert "timed out after 7 seconds" in raw_text
    assert "partial stdout" in raw_text
    assert "partial stderr" in raw_text


def test_main_fail_on_errors_propagates_collection_status(monkeypatch, tmp_path):
    monkeypatch.setattr(
        harvest,
        "build_report",
        lambda: ("report", "non-import collection failure", "env", 4),
    )
    out = tmp_path / "report.md"

    with pytest.raises(SystemExit) as excinfo:
        harvest.main(["--report", str(out), "--fail-on-errors"])

    assert excinfo.value.code == 4
    assert out.read_text(encoding="utf-8") == "report"


def test_main_report_only_allows_collection_status(monkeypatch, tmp_path):
    monkeypatch.setattr(
        harvest,
        "build_report",
        lambda: ("report", "non-import collection failure", "env", 4),
    )
    out = tmp_path / "report.md"

    harvest.main(["--report", str(out)])

    assert out.read_text(encoding="utf-8") == "report"
