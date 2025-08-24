"""Smoke-test the pytest runner's echoed command."""
from __future__ import annotations

import importlib.util as iu
import os
import subprocess
import sys
from pathlib import Path


def _first_echo_line(text: str) -> str:
    for line in text.splitlines():
        idx = line.find("[run_pytest]")
        if idx != -1:
            return line[idx:]
    raise AssertionError("runner did not echo a '[run_pytest]' line; got:\n" + text)


def test_runner_echo_contains_core_flags():
    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    root = Path(__file__).resolve().parents[1]
    script = root / "tools" / "run_pytest.py"
    args = [
        sys.executable,
        str(script),
        "--disable-warnings",
        "--collect-only",
        "-k",
        "__never__",
    ]

    proc = subprocess.run(
        args, capture_output=True, text=True, env=env, cwd=root
    )

    echo = _first_echo_line(proc.stderr)

    assert "pytest -q" in echo
    assert " -W ignore" in echo

    xdist_present = iu.find_spec("xdist") is not None
    if xdist_present:
        assert "xdist.plugin" in echo, echo
        assert " -n " in echo or echo.endswith(" -n")
    else:
        assert "xdist.plugin" not in echo
        assert " -n " not in echo

