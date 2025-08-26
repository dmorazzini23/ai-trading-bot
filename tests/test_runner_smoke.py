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


def test_runner_echo_exact_command(tmp_path):  # AI-AGENT-REF: check exact runner echo
    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    args = [
        sys.executable,
        "tools/run_pytest.py",
        "--disable-warnings",
        "-q",
        "--files",
        "tests/test_utils_timing.py",
        "tests/test_utils_sleep_shadowing.py",
    ]
    proc = subprocess.run(args, capture_output=True, text=True, env=env)
    assert proc.returncode == 0
    echo = _first_echo_line(proc.stderr)
    xdist_present = iu.find_spec("xdist") is not None
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-W",
        "ignore",
    ]
    if xdist_present:
        cmd += ["-p", "xdist.plugin", "-n", "auto"]
    cmd += [
        "tests/test_utils_timing.py",
        "tests/test_utils_sleep_shadowing.py",
    ]
    expected = "[run_pytest] " + " ".join(cmd)
    assert echo.strip() == expected

