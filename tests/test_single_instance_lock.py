import os, tempfile, pytest


@pytest.mark.parametrize("envvar", ["AI_TRADING_RUNTIME_DIR"])
def test_single_instance_lock_no_sys_exit(monkeypatch, envvar):
    # Use a temp runtime dir; no reliance on /tmp state
    d = tempfile.mkdtemp(prefix="ai-trading-test-")
    monkeypatch.setenv(envvar, d)
    from ai_trading.process_manager import ProcessManager

    pm1 = ProcessManager()
    assert pm1.ensure_single_instance(), "first lock acquire failed"

    pm2 = ProcessManager()
    with pytest.raises(Exception):
        # Second acquire should raise/log (not hard-exit) if code was patched accordingly
        pm2.ensure_single_instance()
