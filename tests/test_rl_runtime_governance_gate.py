from __future__ import annotations

import json
from pathlib import Path

from ai_trading.core import bot_engine


def _write_sidecar(path: Path, *, status: str, recommend: bool) -> Path:
    sidecar = Path(f"{path}.governance.json")
    payload = {
        "governance_status": status,
        "recommend_use_rl_agent": recommend,
    }
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    return sidecar


def test_rl_runtime_governance_allows_valid_sidecar(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "rl_agent.zip"
    model_path.write_bytes(b"rl-model")
    _write_sidecar(model_path, status="production", recommend=True)
    monkeypatch.setenv("AI_TRADING_RL_RUNTIME_REQUIRE_GOVERNANCE", "1")
    monkeypatch.setenv("AI_TRADING_RL_RUNTIME_REQUIRE_PRODUCTION", "1")
    monkeypatch.setenv("AI_TRADING_RL_RUNTIME_REQUIRE_RECOMMEND", "1")

    allowed, reason = bot_engine._rl_runtime_governance_allows_load(
        model_path,
        test_mode=False,
    )
    assert allowed is True
    assert reason == "ok"


def test_rl_runtime_governance_blocks_missing_sidecar(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "rl_agent.zip"
    model_path.write_bytes(b"rl-model")
    monkeypatch.setenv("AI_TRADING_RL_RUNTIME_REQUIRE_GOVERNANCE", "1")

    allowed, reason = bot_engine._rl_runtime_governance_allows_load(
        model_path,
        test_mode=False,
    )
    assert allowed is False
    assert reason == "sidecar_missing"
