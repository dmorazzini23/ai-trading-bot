import sys
import pytest

from ai_trading.utils.safe_subprocess import safe_subprocess_run


def test_safe_subprocess_run_success():
    res = safe_subprocess_run([sys.executable, "-c", "print('ok')"])
    assert res.stdout == "ok"
    assert res.returncode == 0
    assert not res.timeout


def test_safe_subprocess_run_timeout(caplog):
    cmd = [sys.executable, "-c", "import time; time.sleep(1)"]
    with caplog.at_level("WARNING"):
        res = safe_subprocess_run(cmd, timeout=0.1)
    assert res.stdout == ""
    assert res.timeout
    assert res.returncode == -1
    assert any("timed out" in rec.message for rec in caplog.records)


def test_safe_subprocess_run_immediate_timeout(caplog):
    cmd = [sys.executable, "-c", "print('never runs')"]
    with caplog.at_level("WARNING"):
        res = safe_subprocess_run(cmd, timeout=0)
    assert res.stdout == ""
    assert res.timeout
    assert res.returncode == -1
    assert any("timed out" in rec.message for rec in caplog.records)


def test_safe_subprocess_run_nonzero_exit(caplog):
    cmd = [sys.executable, "-c", "import sys; sys.exit(2)"]
    with caplog.at_level("WARNING"):
        res = safe_subprocess_run(cmd)
    assert res.stdout == ""
    assert res.returncode == 2
    assert not res.timeout
    assert any(str(cmd) in rec.message for rec in caplog.records)
