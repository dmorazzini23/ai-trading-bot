"""Unit tests for tools.run_pytest command construction."""
from __future__ import annotations

import importlib.util as iu
import os
import sys

import tools.run_pytest as run_pytest


def test_build_pytest_cmd_echo(monkeypatch):  # AI-AGENT-REF: verify exact command
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    parser = run_pytest.build_parser()
    args = parser.parse_args(
        [
            "--disable-warnings",
            "-q",
            "--files",
            "tests/test_utils_timing.py",
            "tests/test_trading_config_aliases.py",
        ]
    )
    cmd = run_pytest.build_pytest_cmd(args)
    expected = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-W",
        "ignore",
    ]
    addopts = os.environ.get("PYTEST_ADDOPTS", "")
    if ("-p pytest_asyncio" not in addopts) and (iu.find_spec("pytest_asyncio") is not None):
        expected += ["-p", "pytest_asyncio.plugin"]
    no_xdist = os.environ.get("NO_XDIST") == "1"
    if ("-p xdist.plugin" not in addopts) and (iu.find_spec("xdist") is not None) and not no_xdist:
        expected += ["-p", "xdist.plugin", "-n", os.environ.get("PYTEST_XDIST_N", "auto")]
    if iu.find_spec("pytest_timeout") is not None:
        expected += ["-p", "pytest_timeout"]
    expected += [
        "tests/test_utils_timing.py",
        "tests/test_trading_config_aliases.py",
    ]
    assert cmd == expected
    assert run_pytest.echo_command(cmd) == "[run_pytest] " + " ".join(expected)

