from __future__ import annotations

import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_script():
    name = "runtime_preflight_check_under_test"
    spec = spec_from_file_location(
        name, PROJECT_ROOT / "scripts" / "runtime_preflight_check.py"
    )
    module = module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_gonogo_failures_are_hard_failures(monkeypatch):
    script = _load_script()

    def raises_timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["python"], timeout=60)

    monkeypatch.setattr(script.subprocess, "run", raises_timeout)
    result = script.check_gonogo(PROJECT_ROOT)
    assert result.status == "fail"
    assert result.summary == "unable to run runtime_gonogo_status"

    monkeypatch.setattr(
        script.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="runtime_gonogo_status produced no payload",
            stderr="",
        ),
    )
    result = script.check_gonogo(PROJECT_ROOT)
    assert result.status == "fail"
    assert result.summary == "no JSON payload parsed from runtime_gonogo_status output"

    monkeypatch.setattr(
        script.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=2,
            stdout='{"gate_passed": true, "failed_checks": []}',
            stderr="tool failed",
        ),
    )
    result = script.check_gonogo(PROJECT_ROOT)
    assert result.status == "fail"
    assert result.summary == "runtime_gonogo_status exited non-zero"

    monkeypatch.setattr(
        script.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout='{"gate_passed": false, "failed_checks": ["slippage"]}',
            stderr="",
        ),
    )
    result = script.check_gonogo(PROJECT_ROOT)
    assert result.status == "fail"
    assert result.summary == "go/no-go FAIL"


def test_stale_optional_context_remains_warning(tmp_path):
    script = _load_script()
    result = script.check_report_freshness(tmp_path)
    assert result.status == "warn"
