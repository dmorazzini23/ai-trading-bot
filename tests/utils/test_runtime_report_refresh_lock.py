from __future__ import annotations

from ai_trading.utils import process_manager


def test_persistent_lock_file_does_not_block_refresh_after_process_exit(tmp_path, monkeypatch):
    lock_path = tmp_path / "replay_governance_refresh.lock"
    lock_path.write_text("previous process exited")
    monkeypatch.setattr(process_manager, "_lock_path", lambda name: lock_path)
    name = "runtime-report-replay-governance"

    assert process_manager.acquire_lock(name, timeout=0) is True
    try:
        assert process_manager.acquire_lock(name, timeout=0) is False
    finally:
        process_manager.release_lock(name)

    assert process_manager.acquire_lock(name, timeout=0) is True
    process_manager.release_lock(name)
