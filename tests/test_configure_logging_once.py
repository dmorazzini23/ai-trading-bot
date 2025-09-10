import importlib
import logging
import os

import ai_trading.logging as L


def test_configure_logging_logs_once(capsys):
    root = logging.getLogger()
    original_handlers = root.handlers.copy()
    try:
        root.handlers.clear()
        os.environ['PYTEST_RUNNING'] = '1'
        log_mod = importlib.reload(L)
        log_mod.configure_logging()
        log_mod.configure_logging()
        out = capsys.readouterr().out
        assert out.count('Logging configured successfully - no duplicates possible') == 1
    finally:
        root.handlers = original_handlers
        importlib.reload(L)
        os.environ.pop('PYTEST_RUNNING', None)
