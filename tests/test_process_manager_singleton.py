"""Tests for ProcessManager single-instance enforcement."""  # AI-AGENT-REF

from __future__ import annotations

from ai_trading.process_manager import ProcessManager


def test_process_manager_single_instance(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_RUNTIME_DIR", str(tmp_path))
    pm1 = ProcessManager(lock_name="unit", dir_env="TEST_RUNTIME_DIR")
    assert pm1.ensure_single_instance() is True
    pm2 = ProcessManager(lock_name="unit", dir_env="TEST_RUNTIME_DIR")
    assert pm2.ensure_single_instance() is False
    pm1._cleanup()

