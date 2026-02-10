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
        "tests/test_trading_config_aliases.py",
    ]
    proc = subprocess.run(args, capture_output=True, text=True, env=env)
    assert proc.returncode == 0
    echo = _first_echo_line(proc.stderr)
    import tools.run_pytest as run_pytest

    cmd = [sys.executable, "-m", "pytest", "-q", "-W", "ignore"]
    addopts = env.get("PYTEST_ADDOPTS", "")
    if (
        env.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") == "1"
        and ("-p pytest_asyncio" not in addopts)
        and (iu.find_spec("pytest_asyncio") is not None)
    ):
        cmd += ["-p", "pytest_asyncio.plugin"]
    if (
        env.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") == "1"
        and ("-p xdist.plugin" not in addopts)
        and (iu.find_spec("xdist") is not None)
        and env.get("NO_XDIST") != "1"
    ):
        cmd += ["-p", "xdist.plugin", "-n", env.get("PYTEST_XDIST_N", "auto")]
    if (
        env.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") == "1"
        and iu.find_spec("pytest_timeout") is not None
    ):
        cmd += ["-p", "pytest_timeout"]
    cmd += [
        "tests/test_utils_timing.py",
        "tests/test_trading_config_aliases.py",
    ]
    expected = run_pytest.echo_command(cmd)
    assert echo.strip() == expected
