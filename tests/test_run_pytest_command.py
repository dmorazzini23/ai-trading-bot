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


def test_build_subprocess_env_scrubs_envfile_keys(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ALPACA_API_KEY=from_file\nTRADING_MODE=balanced\n# comment\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("ALPACA_API_KEY", "exported")
    monkeypatch.setenv("TRADING_MODE", "aggressive")
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))
    monkeypatch.delenv("AI_TRADING_PYTEST_KEEP_ENV", raising=False)

    child_env = run_pytest._build_subprocess_env(tmp_path)

    assert "ALPACA_API_KEY" not in child_env
    assert "TRADING_MODE" not in child_env
    assert child_env["PATH"] == os.environ.get("PATH", "")
    assert child_env["PYTEST_RUNNING"] == "1"


def test_build_subprocess_env_respects_keep_env_opt_out(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("ALPACA_API_KEY=from_file\n", encoding="utf-8")

    monkeypatch.setenv("ALPACA_API_KEY", "exported")
    monkeypatch.setenv("AI_TRADING_PYTEST_KEEP_ENV", "1")

    child_env = run_pytest._build_subprocess_env(tmp_path)

    assert child_env.get("ALPACA_API_KEY") == "exported"
    assert child_env["PYTEST_RUNNING"] == "1"
