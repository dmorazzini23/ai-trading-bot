import importlib
import logging
import os

import ai_trading.logging as L


def test_configure_logging_logs_once(capsys, monkeypatch):
    root = logging.getLogger()
    original_handlers = root.handlers.copy()
    original_test_mode = os.environ.get('PYTEST_RUNNING')
    with monkeypatch.context() as test_env:
        test_env.setenv('PYTEST_RUNNING', '1')
        try:
            root.handlers.clear()
            log_mod = importlib.reload(L)
            log_mod.configure_logging()
            log_mod.configure_logging()
            out = capsys.readouterr().out
            assert out.count('Logging configured successfully - no duplicates possible') == 1
        finally:
            root.handlers = original_handlers
            importlib.reload(L)
    assert os.environ.get('PYTEST_RUNNING') == original_test_mode
