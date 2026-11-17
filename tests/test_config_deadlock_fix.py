"""Tests for configuration validation deadlock fix."""

import os
import subprocess
import sys
import threading
import time
from unittest.mock import patch

import pytest
from ai_trading import config














def test_main_import_no_hang():
    """Test that importing ai_trading.main doesn't hang."""
    start_time = time.time()

    try:
        # This is the specific test case mentioned in the problem statement
        import ai_trading.main  # noqa: F401
    except ImportError:
        # Import might fail due to missing dependencies, but it shouldn't hang
        pass

    elapsed = time.time() - start_time
    assert elapsed < 10.0, f"Import took too long (possible hang): {elapsed:.2f} seconds"


def test_main_import_does_not_initialize_runtime_logging_or_paths():
    """Importing ai_trading.main should not load dotenv or create log paths."""

    code = """
import importlib
import os
import pathlib
import sys
import tempfile

base = pathlib.Path(tempfile.mkdtemp())
log_parent = base / "logs"
os.environ["BOT_LOG_FILE"] = str(log_parent / "bot.log")
os.environ.pop("PYTEST_RUNNING", None)
os.environ.pop("LOG_DISABLE_QUEUE", None)

import ai_trading.logging as logging_mod

calls = []
logging_mod.configure_logging = lambda *args, **kwargs: calls.append("configure_logging")

importlib.import_module("ai_trading.main")

assert calls == [], calls
assert not log_parent.exists(), log_parent
assert getattr(logging_mod, "_LOGGING_LISTENER", None) is None
assert "ai_trading.env" not in sys.modules
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr


