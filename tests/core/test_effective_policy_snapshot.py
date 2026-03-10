from __future__ import annotations

import json
from pathlib import Path

from ai_trading.core import bot_engine as eng


class _DummyPolicy:
    policy_hash = "policy-hash-123"

    def to_dict(self) -> dict[str, object]:
        return {"risk": {"max_position_pct": 0.08}}


def test_effective_policy_snapshot_falls_back_on_permission_error(
    monkeypatch,
    tmp_path: Path,
) -> None:
    preferred_path = tmp_path / "runtime" / "effective_policy.json"
    fallback_path = tmp_path / "fallback" / "effective_policy.json"

    monkeypatch.setattr(
        eng,
        "resolve_runtime_artifact_path",
        lambda *_args, **_kwargs: preferred_path,
    )
    monkeypatch.setattr(
        eng,
        "_compute_user_state_trade_log_path",
        lambda _filename="effective_policy.json": str(fallback_path),
    )

    original_write_text = Path.write_text

    def _deny_preferred_write(path: Path, payload: str, *args, **kwargs) -> int:
        if path == preferred_path:
            raise PermissionError("permission denied")
        return original_write_text(path, payload, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", _deny_preferred_write)

    state = eng.BotState()
    eng._persist_effective_policy_snapshot(state, _DummyPolicy(), loop_id="loop-1")

    assert state.effective_policy_snapshot_path == str(fallback_path)
    payload = json.loads(fallback_path.read_text(encoding="utf-8"))
    assert payload["loop_id"] == "loop-1"
    assert payload["policy_hash"] == "policy-hash-123"


def test_effective_policy_snapshot_repairs_permission_with_unlink(
    monkeypatch,
    tmp_path: Path,
) -> None:
    preferred_path = tmp_path / "runtime" / "effective_policy.json"
    fallback_path = tmp_path / "fallback" / "effective_policy.json"
    preferred_path.parent.mkdir(parents=True, exist_ok=True)
    preferred_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        eng,
        "resolve_runtime_artifact_path",
        lambda *_args, **_kwargs: preferred_path,
    )
    monkeypatch.setattr(
        eng,
        "_compute_user_state_trade_log_path",
        lambda _filename="effective_policy.json": str(fallback_path),
    )

    original_write_text = Path.write_text
    original_unlink = Path.unlink
    write_attempts = {"preferred": 0}
    unlink_attempts = {"preferred": 0}

    def _fail_once_then_write(path: Path, payload: str, *args, **kwargs) -> int:
        if path == preferred_path and write_attempts["preferred"] == 0:
            write_attempts["preferred"] += 1
            raise PermissionError("permission denied")
        if path == preferred_path:
            write_attempts["preferred"] += 1
        return original_write_text(path, payload, *args, **kwargs)

    def _track_unlink(path: Path, *args, **kwargs) -> None:
        if path == preferred_path:
            unlink_attempts["preferred"] += 1
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", _fail_once_then_write)
    monkeypatch.setattr(Path, "unlink", _track_unlink)

    state = eng.BotState()
    eng._persist_effective_policy_snapshot(state, _DummyPolicy(), loop_id="loop-2")

    assert unlink_attempts["preferred"] == 1
    assert state.effective_policy_snapshot_path == str(preferred_path)
    assert not fallback_path.exists()
    payload = json.loads(preferred_path.read_text(encoding="utf-8"))
    assert payload["loop_id"] == "loop-2"
    assert payload["policy_hash"] == "policy-hash-123"
