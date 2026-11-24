from __future__ import annotations

import importlib
import sys

import pytest


def test_apca_env_rejects_legacy_keys(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "x")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "y")
    module_name = "ai_trading.config.runtime"
    # Restore the original module afterwards: a re-import would leave a second
    # TradingConfig class that isinstance checks elsewhere no longer accept.
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    with pytest.raises(RuntimeError, match="Legacy APCA_\\* environment variables"):
        importlib.import_module(module_name)


def test_env_doctor_detects_process_env(monkeypatch):
    monkeypatch.setenv("APCA_FAKE", "1")
    import tools.env_doctor as env_doctor

    exit_code = env_doctor.main()
    assert exit_code != 0
