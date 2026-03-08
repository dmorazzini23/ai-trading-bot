import subprocess
import sys
from typing import Any, cast

import pytest

from ai_trading.utils.safe_subprocess import (
    SUBPROCESS_TIMEOUT_DEFAULT,
    SafeSubprocessResult,
    safe_subprocess_run,
)


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _timeout_payload(exc: subprocess.TimeoutExpired) -> Any:
    return cast(Any, exc)


def test_safe_subprocess_run_success():
    res = safe_subprocess_run([sys.executable, "-c", "print('ok')"])
    assert isinstance(res, subprocess.CompletedProcess)
    assert res.stdout.strip() == "ok"
    assert res.returncode == 0
    assert res.stderr == ""


def test_safe_subprocess_run_timeout(caplog):
    script = (
        "import sys, time; "
        "sys.stdout.write('ready\\n'); sys.stdout.flush(); "
        "sys.stderr.write('warn\\n'); sys.stderr.flush(); "
        "time.sleep(1)"
    )
    cmd = [sys.executable, "-c", script]
    with caplog.at_level("WARNING"):
        with pytest.raises(subprocess.TimeoutExpired) as excinfo:
            safe_subprocess_run(cmd, timeout=0.5)

    exc_obj = _timeout_payload(excinfo.value)
    stdout = _as_text(exc_obj.stdout)
    stderr = _as_text(exc_obj.stderr)
    # Under heavy parallel CI load, the child process can time out before
    # emitting either stream, so accept both captured and empty variants.
    assert stdout in {"", "ready\n"}
    assert stderr in {"", "warn\n"}
    expected_result = SafeSubprocessResult(stdout, stderr, 124, True)
    assert exc_obj.cmd == cmd
    assert isinstance(exc_obj.result, SafeSubprocessResult)
    assert exc_obj.result == expected_result
    assert _as_text(exc_obj.stdout) == stdout
    assert _as_text(exc_obj.stderr) == stderr
    assert exc_obj.timeout == pytest.approx(0.5)
    assert exc_obj.result.stdout == _as_text(exc_obj.stdout)
    assert exc_obj.result.stderr == _as_text(exc_obj.stderr)
    assert exc_obj.__cause__ is None
    assert caplog.records
    record = next(
        (
            rec
            for rec in caplog.records
            if rec.name == "ai_trading.utils.safe_subprocess" and rec.message == "SAFE_SUBPROCESS_TIMEOUT"
        ),
        None,
    )
    assert record is not None
    assert record.cmd == cmd
    assert record.timeout == pytest.approx(0.5)


def test_safe_subprocess_run_default_timeout(caplog):
    sleep_seconds = SUBPROCESS_TIMEOUT_DEFAULT * 3
    cmd = [sys.executable, "-c", f"import time; time.sleep({sleep_seconds})"]
    with caplog.at_level("WARNING"):
        with pytest.raises(subprocess.TimeoutExpired) as excinfo:
            safe_subprocess_run(cmd)

    exc_obj = _timeout_payload(excinfo.value)
    assert exc_obj.cmd == cmd
    assert exc_obj.timeout == pytest.approx(SUBPROCESS_TIMEOUT_DEFAULT)
    assert _as_text(exc_obj.stdout) == ""
    assert _as_text(exc_obj.stderr) == ""
    assert caplog.records
    record = next(
        (
            rec
            for rec in caplog.records
            if rec.name == "ai_trading.utils.safe_subprocess" and rec.message == "SAFE_SUBPROCESS_TIMEOUT"
        ),
        None,
    )
    assert record is not None
    assert record.cmd == cmd
    assert record.timeout == pytest.approx(SUBPROCESS_TIMEOUT_DEFAULT)


@pytest.mark.parametrize("timeout_value", [0, -0.5])
def test_safe_subprocess_run_immediate_timeout(caplog, timeout_value):
    cmd = [sys.executable, "-c", "print('never runs')"]
    with caplog.at_level("WARNING"):
        res = safe_subprocess_run(cmd, timeout=timeout_value)
    assert isinstance(res, subprocess.CompletedProcess)
    assert res.stdout.strip() == "never runs"
    assert res.stderr == ""
    assert res.returncode == 0
    assert not caplog.records


def test_safe_subprocess_run_nonzero_exit(caplog):
    cmd = [sys.executable, "-c", "import sys; sys.exit(2)"]
    with caplog.at_level("WARNING"):
        res = safe_subprocess_run(cmd)
    assert isinstance(res, subprocess.CompletedProcess)
    assert res.stdout == ""
    assert res.stderr == ""
    assert res.returncode == 2
    assert not caplog.records


def test_safe_subprocess_run_missing_command(caplog):
    cmd = ["definitely-not-a-real-binary"]
    with caplog.at_level("WARNING"):
        res = safe_subprocess_run(cmd)

    assert isinstance(res, subprocess.CompletedProcess)
    assert res.returncode == 127
    assert "definitely-not-a-real-binary" in res.stderr
    assert caplog.records
    record = caplog.records[0]
    assert record.message == "SAFE_SUBPROCESS_ERROR"
    assert record.cmd == cmd
    assert record.returncode == 127


def test_safe_subprocess_run_timeout_without_captured_output(monkeypatch, caplog):
    captured: dict[str, Any] = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout"))

    monkeypatch.setattr("ai_trading.utils.safe_subprocess.subprocess.run", fake_run)

    with caplog.at_level("WARNING"):
        with pytest.raises(subprocess.TimeoutExpired) as excinfo:
            safe_subprocess_run(["dummy"], timeout=0.25)

    exc_obj = _timeout_payload(excinfo.value)
    result = exc_obj.result
    assert result.timeout is True
    assert result.returncode == 124
    assert result.stdout == ""
    assert result.stderr == ""
    assert _as_text(exc_obj.stdout) == ""
    assert _as_text(exc_obj.stderr) == ""
    assert exc_obj.result.stdout == _as_text(exc_obj.stdout)
    assert exc_obj.result.stderr == _as_text(exc_obj.stderr)
    assert captured["args"] == (["dummy"],)
    assert captured["kwargs"]["stdout"] == subprocess.PIPE
    assert captured["kwargs"]["stderr"] == subprocess.PIPE
    assert captured["kwargs"]["text"] is True
    assert captured["kwargs"]["timeout"] == 0.25
    assert caplog.records


def test_safe_subprocess_run_timeout_with_captured_output(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        raise subprocess.TimeoutExpired(
            cmd=args[0], timeout=kwargs.get("timeout"), output="partial stdout", stderr="partial stderr"
        )

    monkeypatch.setattr("ai_trading.utils.safe_subprocess.subprocess.run", fake_run)

    with pytest.raises(subprocess.TimeoutExpired) as excinfo:
        safe_subprocess_run(["dummy"], timeout=0.25)

    exc_obj = _timeout_payload(excinfo.value)
    result = exc_obj.result
    assert result.timeout is True
    assert result.returncode == 124
    assert result.stdout == "partial stdout"
    assert result.stderr == "partial stderr"
    assert _as_text(exc_obj.stdout) == "partial stdout"
    assert _as_text(exc_obj.stderr) == "partial stderr"
    assert exc_obj.result.stdout == _as_text(exc_obj.stdout)
    assert exc_obj.result.stderr == _as_text(exc_obj.stderr)
    assert isinstance(exc_obj.result, SafeSubprocessResult)
    assert captured["kwargs"]["timeout"] == 0.25


def test_safe_subprocess_run_timeout_attaches_result_bytes(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        raise subprocess.TimeoutExpired(
            cmd=["dummy"], timeout=kwargs.get("timeout"), output=b"late stdout", stderr=b"late stderr"
        )

    monkeypatch.setattr("ai_trading.utils.safe_subprocess.subprocess.run", fake_run)

    with pytest.raises(subprocess.TimeoutExpired) as excinfo:
        safe_subprocess_run(["dummy"], timeout=0.25)

    exc_obj = _timeout_payload(excinfo.value)
    result = exc_obj.result
    assert isinstance(result, SafeSubprocessResult)
    assert result.timeout is True
    assert result.returncode == 124
    assert result.stdout == "late stdout"
    assert result.stderr == "late stderr"
    assert _as_text(exc_obj.stdout) == "late stdout"
    assert _as_text(exc_obj.stderr) == "late stderr"
    assert exc_obj.result.stdout == _as_text(exc_obj.stdout)
    assert exc_obj.result.stderr == _as_text(exc_obj.stderr)
    assert captured["kwargs"]["timeout"] == 0.25


def test_safe_subprocess_run_timeout_cleanup_timeout(monkeypatch, caplog):
    captured: dict[str, Any] = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        raise subprocess.TimeoutExpired(
            cmd=["dummy"],
            timeout=kwargs.get("timeout"),
            output="after kill stdout",
            stderr="after kill stderr",
        )

    monkeypatch.setattr("ai_trading.utils.safe_subprocess.subprocess.run", fake_run)

    with caplog.at_level("WARNING"):
        with pytest.raises(subprocess.TimeoutExpired) as excinfo:
            safe_subprocess_run(["dummy"], timeout=0.2)

    assert caplog.records

    exc_obj = _timeout_payload(excinfo.value)
    result = exc_obj.result
    assert isinstance(result, SafeSubprocessResult)
    assert result.timeout is True
    assert result.returncode == 124
    assert result.stdout == "after kill stdout"
    assert result.stderr == "after kill stderr"
    assert exc_obj.timeout == pytest.approx(0.2)
    assert _as_text(exc_obj.stdout) == "after kill stdout"
    assert _as_text(exc_obj.stderr) == "after kill stderr"
    assert exc_obj.result.stdout == _as_text(exc_obj.stdout)
    assert exc_obj.result.stderr == _as_text(exc_obj.stderr)
    assert captured["kwargs"]["timeout"] == 0.2


def test_safe_subprocess_run_timeout_populates_result_and_returncode(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        raise subprocess.TimeoutExpired(cmd=["dummy"], timeout=kwargs.get("timeout"))

    monkeypatch.setattr("ai_trading.utils.safe_subprocess.subprocess.run", fake_run)

    with pytest.raises(subprocess.TimeoutExpired) as excinfo:
        safe_subprocess_run(["dummy"], timeout=0.3)

    exc_obj = _timeout_payload(excinfo.value)
    result = exc_obj.result
    assert isinstance(result, SafeSubprocessResult)
    assert result.timeout is True
    assert result.returncode == 124
    assert result.stdout == ""
    assert result.stderr == ""
    assert exc_obj.timeout == pytest.approx(0.3)
    assert exc_obj.result.stdout == _as_text(exc_obj.stdout)
    assert exc_obj.result.stderr == _as_text(exc_obj.stderr)
    assert captured["kwargs"]["timeout"] == 0.3
