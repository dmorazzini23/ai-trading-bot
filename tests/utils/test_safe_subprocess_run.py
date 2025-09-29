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


def test_safe_subprocess_run_immediate_timeout(caplog):
    cmd = [sys.executable, "-c", "print('never runs')"]
    with caplog.at_level("WARNING"):
        res = safe_subprocess_run(cmd, timeout=0)
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

    class DummyProcess:
        def __init__(self) -> None:
            self.kill_called = False
            self.communicate_calls: list[float | None] = []
            self.returncode = None

        def communicate(self, timeout: float | None = None):
            self.communicate_calls.append(timeout)
            if timeout is not None:
                raise subprocess.TimeoutExpired(cmd=["dummy"], timeout=timeout)
            return None, None

        def kill(self) -> None:
            self.kill_called = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    dummy_proc = DummyProcess()

    def fake_popen(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return dummy_proc

    monkeypatch.setattr("ai_trading.utils.safe_subprocess.subprocess.Popen", fake_popen)

    with caplog.at_level("WARNING"):
        res = safe_subprocess_run(["dummy"], timeout=0.25)

    assert res.timeout is True
    assert res.returncode == 124
    assert res.stdout == ""
    assert res.stderr == ""
    assert dummy_proc.kill_called is True
    assert dummy_proc.communicate_calls == [0.25, None]
    assert calls["kwargs"]["stdout"] == subprocess.PIPE
    assert calls["kwargs"]["stderr"] == subprocess.PIPE
    assert calls["kwargs"]["text"] is True
    assert not caplog.records
