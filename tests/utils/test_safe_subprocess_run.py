import sys
import subprocess

import pytest

from ai_trading.utils.safe_subprocess import safe_subprocess_run


def test_safe_subprocess_run_success():
    res = safe_subprocess_run([sys.executable, "-c", "print('ok')"])
    assert res.stdout.strip() == "ok"
    assert res.returncode == 0
    assert res.stderr == ""
    assert not res.timeout


def test_safe_subprocess_run_timeout(caplog):
    cmd = [sys.executable, "-c", "import time; time.sleep(1)"]
    with caplog.at_level("WARNING"):
        res = safe_subprocess_run(cmd, timeout=0.1)
    assert res.stdout == ""
    assert res.stderr == ""
    assert res.timeout
    assert res.returncode == 124
    assert not caplog.records  # timeout should not emit warnings


@pytest.mark.parametrize("timeout_value", [0, -0.5])
def test_safe_subprocess_run_immediate_timeout(caplog, timeout_value):
    cmd = [sys.executable, "-c", "print('never runs')"]
    with caplog.at_level("WARNING"):
        res = safe_subprocess_run(cmd, timeout=timeout_value)
    assert res.stdout == ""
    assert res.stderr == ""
    assert res.timeout
    assert res.returncode == -1
    assert any("timed out" in rec.message for rec in caplog.records)


def test_safe_subprocess_run_nonzero_exit(caplog):
    cmd = [sys.executable, "-c", "import sys; sys.exit(2)"]
    with caplog.at_level("WARNING"):
        res = safe_subprocess_run(cmd)
    assert res.stdout == ""
    assert res.stderr == ""
    assert res.returncode == 2
    assert not res.timeout
    assert not caplog.records


def test_safe_subprocess_run_timeout_without_captured_output(monkeypatch, caplog):
    calls: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        exc = subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])
        # ``subprocess.run`` populates ``stdout``/``stderr`` attributes; mimic that here.
        exc.stdout = None
        exc.stderr = None
        raise exc

    monkeypatch.setattr("ai_trading.utils.safe_subprocess.subprocess.run", fake_run)

    with caplog.at_level("WARNING"):
        res = safe_subprocess_run(["dummy"], timeout=0.25)

    assert res.timeout is True
    assert res.returncode == 124
    assert res.stdout == ""
    assert res.stderr == ""
    assert calls["args"] == (["dummy"],)
    assert calls["kwargs"]["stdout"] == subprocess.PIPE
    assert calls["kwargs"]["stderr"] == subprocess.PIPE
    assert calls["kwargs"]["text"] is True
    assert calls["kwargs"]["check"] is False
    assert calls["kwargs"]["timeout"] == 0.25
    assert not caplog.records


def test_safe_subprocess_run_timeout_with_captured_output(monkeypatch):
    def fake_run(*args, **kwargs):
        exc = subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])
        exc.stdout = "partial stdout"
        exc.stderr = "partial stderr"
        raise exc

    monkeypatch.setattr("ai_trading.utils.safe_subprocess.subprocess.run", fake_run)

    res = safe_subprocess_run(["dummy"], timeout=0.25)

    assert res.timeout is True
    assert res.returncode == 124
    assert res.stdout == "partial stdout"
    assert res.stderr == "partial stderr"
