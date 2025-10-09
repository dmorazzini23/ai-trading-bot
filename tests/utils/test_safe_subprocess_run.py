import subprocess
import sys

import pytest

from ai_trading.utils.safe_subprocess import SafeSubprocessResult, safe_subprocess_run


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
            safe_subprocess_run(cmd, timeout=0.3)

    expected_result = SafeSubprocessResult("ready\n", "warn\n", 124, True)
    assert excinfo.value.cmd == cmd
    assert isinstance(excinfo.value.result, SafeSubprocessResult)
    assert excinfo.value.result == expected_result
    assert excinfo.value.stdout == "ready\n"
    assert excinfo.value.stderr == "warn\n"
    assert excinfo.value.timeout == pytest.approx(0.3)
    assert excinfo.value.result.stdout == excinfo.value.stdout
    assert excinfo.value.result.stderr == excinfo.value.stderr
    assert excinfo.value.__cause__ is None
    assert caplog.records
    record = caplog.records[0]
    assert record.message == "SAFE_SUBPROCESS_TIMEOUT"
    assert record.cmd == cmd
    assert record.timeout == pytest.approx(0.3)


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
    captured: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout"))

    monkeypatch.setattr("ai_trading.utils.safe_subprocess.subprocess.run", fake_run)

    with caplog.at_level("WARNING"):
        with pytest.raises(subprocess.TimeoutExpired) as excinfo:
            safe_subprocess_run(["dummy"], timeout=0.25)

    result = excinfo.value.result
    assert result.timeout is True
    assert result.returncode == 124
    assert result.stdout == ""
    assert result.stderr == ""
    assert excinfo.value.stdout == ""
    assert excinfo.value.stderr == ""
    assert excinfo.value.result.stdout == excinfo.value.stdout
    assert excinfo.value.result.stderr == excinfo.value.stderr
    assert captured["args"] == (["dummy"],)
    assert captured["kwargs"]["stdout"] == subprocess.PIPE
    assert captured["kwargs"]["stderr"] == subprocess.PIPE
    assert captured["kwargs"]["text"] is True
    assert captured["kwargs"]["timeout"] == 0.25
    assert caplog.records


def test_safe_subprocess_run_timeout_with_captured_output(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        raise subprocess.TimeoutExpired(
            cmd=args[0], timeout=kwargs.get("timeout"), output="partial stdout", stderr="partial stderr"
        )

    monkeypatch.setattr("ai_trading.utils.safe_subprocess.subprocess.run", fake_run)

    with pytest.raises(subprocess.TimeoutExpired) as excinfo:
        safe_subprocess_run(["dummy"], timeout=0.25)

    result = excinfo.value.result
    assert result.timeout is True
    assert result.returncode == 124
    assert result.stdout == "partial stdout"
    assert result.stderr == "partial stderr"
    assert excinfo.value.stdout == "partial stdout"
    assert excinfo.value.stderr == "partial stderr"
    assert excinfo.value.result.stdout == excinfo.value.stdout
    assert excinfo.value.result.stderr == excinfo.value.stderr
    assert isinstance(excinfo.value.result, SafeSubprocessResult)
    assert captured["kwargs"]["timeout"] == 0.25


def test_safe_subprocess_run_timeout_attaches_result_bytes(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        raise subprocess.TimeoutExpired(
            cmd=["dummy"], timeout=kwargs.get("timeout"), output=b"late stdout", stderr=b"late stderr"
        )

    monkeypatch.setattr("ai_trading.utils.safe_subprocess.subprocess.run", fake_run)

    with pytest.raises(subprocess.TimeoutExpired) as excinfo:
        safe_subprocess_run(["dummy"], timeout=0.25)

    result = excinfo.value.result
    assert isinstance(result, SafeSubprocessResult)
    assert result.timeout is True
    assert result.returncode == 124
    assert result.stdout == "late stdout"
    assert result.stderr == "late stderr"
    assert excinfo.value.stdout == "late stdout"
    assert excinfo.value.stderr == "late stderr"
    assert excinfo.value.result.stdout == excinfo.value.stdout
    assert excinfo.value.result.stderr == excinfo.value.stderr
    assert captured["kwargs"]["timeout"] == 0.25


def test_safe_subprocess_run_timeout_cleanup_timeout(monkeypatch, caplog):
    captured: dict[str, object] = {}

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

    result = excinfo.value.result
    assert isinstance(result, SafeSubprocessResult)
    assert result.timeout is True
    assert result.returncode == 124
    assert result.stdout == "after kill stdout"
    assert result.stderr == "after kill stderr"
    assert excinfo.value.timeout == pytest.approx(0.2)
    assert excinfo.value.stdout == "after kill stdout"
    assert excinfo.value.stderr == "after kill stderr"
    assert excinfo.value.result.stdout == excinfo.value.stdout
    assert excinfo.value.result.stderr == excinfo.value.stderr
    assert captured["kwargs"]["timeout"] == 0.2


def test_safe_subprocess_run_timeout_populates_result_and_returncode(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        raise subprocess.TimeoutExpired(cmd=["dummy"], timeout=kwargs.get("timeout"))

    monkeypatch.setattr("ai_trading.utils.safe_subprocess.subprocess.run", fake_run)

    with pytest.raises(subprocess.TimeoutExpired) as excinfo:
        safe_subprocess_run(["dummy"], timeout=0.3)

    result = excinfo.value.result
    assert isinstance(result, SafeSubprocessResult)
    assert result.timeout is True
    assert result.returncode == 124
    assert result.stdout == ""
    assert result.stderr == ""
    assert excinfo.value.timeout == pytest.approx(0.3)
    assert excinfo.value.result.stdout == excinfo.value.stdout
    assert excinfo.value.result.stderr == excinfo.value.stderr
    assert captured["kwargs"]["timeout"] == 0.3
