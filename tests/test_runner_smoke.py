"""Smoke-test the pytest runner's echoed command."""
from __future__ import annotations

import importlib.util as iu
import os
import subprocess
import sys


def _first_echo_line(text: str) -> str:
    for line in text.splitlines():
        idx = line.find("[run_pytest]")
        if idx != -1:
            return line[idx:]
    raise AssertionError("runner did not echo a '[run_pytest]' line; got:\n" + text)


def test_runner_echo_contains_core_flags(tmp_path):
    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    args = [
        sys.executable,
        "tools/run_pytest.py",
        "--disable-warnings",
        "--collect-only",
        "tests/test_utils_timing.py",
    ]
    proc = subprocess.run(args, capture_output=True, text=True, env=env)
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

