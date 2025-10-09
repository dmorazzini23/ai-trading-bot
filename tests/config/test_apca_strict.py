from __future__ import annotations

import importlib
import sys

import pytest


def test_apca_env_allows_legacy_keys(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "x")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "y")
    module_name = "ai_trading.config.runtime"
    if module_name in sys.modules:
        del sys.modules[module_name]
    module = importlib.import_module(module_name)
    assert module is not None


def test_env_doctor_detects_process_env(monkeypatch):
    monkeypatch.setenv("APCA_FAKE", "1")
    import tools.env_doctor as env_doctor

    exit_code = env_doctor.main()
    assert exit_code != 0
