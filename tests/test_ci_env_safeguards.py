import os
import sys
from unittest import mock

import ai_trading.utils.exec as exec_utils
import pytest


def test_ci_env_safeguards(monkeypatch):
    """CI env should force offline, paper-mode tests."""
    monkeypatch.setenv("AI_TRADING_OFFLINE_TESTS", "1")
    monkeypatch.setenv("ALPACA_ENV", "paper")
    env = exec_utils._sanitize_executor_env()
    assert env.get("AI_TRADING_OFFLINE_TESTS") == "1"
    assert os.getenv("ALPACA_ENV") == "paper"
    # Default test fixtures mask API keys with dummy values
    assert os.getenv("ALPACA_API_KEY") == "dummy"
    assert os.getenv("ALPACA_SECRET_KEY") == "dummy"


@pytest.mark.no_test_credentials
def test_no_test_credentials_marker_removes_global_defaults():
    assert os.getenv("ALPACA_API_KEY") is None
    assert os.getenv("ALPACA_SECRET_KEY") is None


def test_sys_modules_clear_patch_preserves_snapshot_modules():
    with mock.patch.dict(sys.modules, {}, clear=True):
        assert sys.modules.get("sys") is sys
        assert "importlib" in sys.modules
