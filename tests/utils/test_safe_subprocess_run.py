import sys
import pytest

from ai_trading.utils.base import safe_subprocess_run


def test_safe_subprocess_run_success():
    out = safe_subprocess_run([sys.executable, "-c", "print('ok')"])
    assert out == "ok"


def test_safe_subprocess_run_timeout(caplog):
    cmd = [sys.executable, "-c", "import time; time.sleep(1)"]
    with caplog.at_level("WARNING"):
        out = safe_subprocess_run(cmd, timeout=0.1)
    assert out == ""
    assert any(str(cmd) in rec.message for rec in caplog.records)


def test_safe_subprocess_run_nonzero_exit(caplog):
    cmd = [sys.executable, "-c", "import sys; sys.exit(2)"]
    with caplog.at_level("WARNING"):
        out = safe_subprocess_run(cmd)
    assert out == ""
    assert any(str(cmd) in rec.message for rec in caplog.records)
