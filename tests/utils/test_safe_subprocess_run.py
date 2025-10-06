import subprocess
import sys

import pytest

from ai_trading.utils.safe_subprocess import SafeSubprocessResult, safe_subprocess_run


def test_safe_subprocess_run_success():
    res = safe_subprocess_run([sys.executable, "-c", "print('ok')"])
    assert res.stdout.strip() == "ok"
    assert res.returncode == 0
    assert res.stderr == ""
    assert not res.timeout


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
    calls: dict[str, object] = {"communicate_calls": [], "wait_calls": []}

    class FakeProc:
        def __init__(self, *args, **kwargs):
            calls["args"] = args
            calls["kwargs"] = kwargs
            self._killed = False
            self.returncode = None
            calls["instance"] = self

        def communicate(self, timeout=None):
            calls["communicate_calls"].append(timeout)
            if timeout is not None and not self._killed:
                raise subprocess.TimeoutExpired(cmd=calls["args"][0], timeout=timeout)
            self.returncode = 124
            return "", ""

        def kill(self):
            self._killed = True
            calls["killed"] = True

        def wait(self, timeout=None):
            calls["wait_calls"].append(timeout)
            if timeout == 0:
                raise subprocess.TimeoutExpired(cmd=calls["args"][0], timeout=timeout)
            self.returncode = 124
            return 124

    monkeypatch.setattr("ai_trading.utils.safe_subprocess.subprocess.Popen", FakeProc)

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
    assert calls["args"] == (["dummy"],)
    assert calls["kwargs"]["stdout"] == subprocess.PIPE
    assert calls["kwargs"]["stderr"] == subprocess.PIPE
    assert calls["kwargs"]["text"] is True
    assert calls.get("killed") is True
    assert calls["instance"]._killed is True
    assert calls["communicate_calls"] == [0.25, 0]
    assert calls["wait_calls"] == [0.25]
    assert not caplog.records


def test_safe_subprocess_run_timeout_with_captured_output(monkeypatch):
    state: dict[str, object] = {"communicate_calls": [], "wait_calls": []}

    class FakeProc:
        def __init__(self, *args, **kwargs):
            self.returncode = None
            self._killed = False
            self.args = args
            state["instance"] = self

        def communicate(self, timeout=None):
            state.setdefault("communicate_calls", []).append(timeout)
            if timeout is not None and not self._killed:
                raise subprocess.TimeoutExpired(
                    cmd=self.args[0],
                    timeout=timeout,
                    output="partial stdout",
                    stderr="partial stderr",
                )
            self.returncode = 124
            return "partial stdout", "partial stderr"

        def kill(self):
            self._killed = True

        def wait(self, timeout=None):
            state.setdefault("wait_calls", []).append(timeout)
            if timeout == 0:
                raise subprocess.TimeoutExpired(cmd=self.args[0], timeout=timeout)
            self.returncode = 124
            return 124

    monkeypatch.setattr("ai_trading.utils.safe_subprocess.subprocess.Popen", FakeProc)

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
    assert state["instance"]._killed is True
    assert state["communicate_calls"] == [0.25, 0]
    assert state["wait_calls"] == [0.25]


def test_safe_subprocess_run_timeout_attaches_result_bytes(monkeypatch):
    state: dict[str, object] = {"communicate_calls": [], "wait_calls": []}

    class FakeProc:
        def __init__(self, *args, **kwargs):
            state["init_args"] = args
            state["init_kwargs"] = kwargs
            self._killed = False
            self.returncode = None
            state["instance"] = self

        def communicate(self, timeout=None):
            state["communicate_calls"].append(timeout)
            if timeout is not None and not self._killed:
                raise subprocess.TimeoutExpired(
                    cmd=["dummy"],
                    timeout=timeout,
                    output=b"late stdout",
                    stderr=b"late stderr",
                )
            self.returncode = 124
            return b"late stdout", b"late stderr"

        def kill(self):
            self._killed = True

        def wait(self, timeout=None):
            state.setdefault("wait_calls", []).append(timeout)
            if timeout == 0:
                raise subprocess.TimeoutExpired(cmd=["dummy"], timeout=timeout)
            self.returncode = 124
            return 124

    monkeypatch.setattr("ai_trading.utils.safe_subprocess.subprocess.Popen", FakeProc)

    with pytest.raises(subprocess.TimeoutExpired) as excinfo:
        safe_subprocess_run(["dummy"], timeout=0.1)

    assert state["communicate_calls"] == [0.1, 0]
    assert state["wait_calls"] == [0.1]
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
    assert state["init_kwargs"]["stdout"] == subprocess.PIPE
    assert state["init_kwargs"]["stderr"] == subprocess.PIPE
    assert state["init_kwargs"]["text"] is True
    assert state["instance"]._killed is True


def test_safe_subprocess_run_timeout_cleanup_timeout(monkeypatch, caplog):
    state: dict[str, object] = {"communicate_calls": [], "wait_calls": []}

    class FakeProc:
        def __init__(self, *args, **kwargs):
            state["init_args"] = args
            state["init_kwargs"] = kwargs
            self._killed = False
            self.returncode = None
            state["instance"] = self

        def communicate(self, timeout=None):
            state["communicate_calls"].append(timeout)
            if len(state["communicate_calls"]) == 1:
                raise subprocess.TimeoutExpired(
                    cmd=["dummy"],
                    timeout=timeout,
                    output="",
                    stderr="",
                )
            raise subprocess.TimeoutExpired(
                cmd=["dummy"],
                timeout=timeout,
                output="after kill stdout",
                stderr="after kill stderr",
            )

        def kill(self):
            self._killed = True
            state["killed"] = True

        def wait(self, timeout=None):
            state.setdefault("wait_calls", []).append(timeout)
            self.returncode = 124
            return 124

    monkeypatch.setattr("ai_trading.utils.safe_subprocess.subprocess.Popen", FakeProc)

    with caplog.at_level("WARNING"):
        with pytest.raises(subprocess.TimeoutExpired) as excinfo:
            safe_subprocess_run(["dummy"], timeout=0.2)

    assert not caplog.records
    assert state["communicate_calls"] == [0.2, 0]
    assert state["wait_calls"] == [0.2]
    assert state.get("killed") is True

    result = excinfo.value.result
    assert isinstance(result, SafeSubprocessResult)
    assert result.timeout is True
    assert result.returncode == 124
    assert result.stdout == "after kill stdout"
    assert result.stderr == "after kill stderr"
    assert excinfo.value.stdout == "after kill stdout"
    assert excinfo.value.stderr == "after kill stderr"
    assert excinfo.value.result.stdout == excinfo.value.stdout
    assert excinfo.value.result.stderr == excinfo.value.stderr
    assert state["init_kwargs"]["stdout"] == subprocess.PIPE
    assert state["init_kwargs"]["stderr"] == subprocess.PIPE
    assert state["init_kwargs"]["text"] is True


def test_safe_subprocess_run_timeout_populates_result_and_returncode(monkeypatch):
    state: dict[str, object] = {"communicate_calls": [], "wait_calls": []}

    class FakeProc:
        def __init__(self, *args, **kwargs):
            state["init_args"] = args
            state["init_kwargs"] = kwargs
            self.returncode = None
            self._killed = False
            state["instance"] = self

        def communicate(self, timeout=None):
            state["communicate_calls"].append(timeout)
            if timeout is not None and not self._killed:
                raise subprocess.TimeoutExpired(cmd=["dummy"], timeout=timeout)
            return "", ""

        def kill(self):
            self._killed = True

        def wait(self, timeout=None):
            state.setdefault("wait_calls", []).append(timeout)
            if timeout == 0:
                raise subprocess.TimeoutExpired(cmd=["dummy"], timeout=timeout)
            self.returncode = 124
            return 124

    monkeypatch.setattr("ai_trading.utils.safe_subprocess.subprocess.Popen", FakeProc)

    with pytest.raises(subprocess.TimeoutExpired) as excinfo:
        safe_subprocess_run(["dummy"], timeout=0.3)

    assert state["communicate_calls"] == [0.3, 0]
    assert state["wait_calls"] == [0.3]
    proc_instance = state["instance"]
    assert proc_instance._killed is True
    assert proc_instance.returncode == 124

    result = excinfo.value.result
    assert isinstance(result, SafeSubprocessResult)
    assert result.timeout is True
    assert result.returncode == 124
    assert result.stdout == ""
    assert result.stderr == ""
    assert excinfo.value.timeout == pytest.approx(0.3)
    assert excinfo.value.result.stdout == excinfo.value.stdout
    assert excinfo.value.result.stderr == excinfo.value.stderr
