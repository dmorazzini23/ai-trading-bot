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


def test_reload_rl_agent_bootstraps_missing_governance_sidecar(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "rl_agent.zip"
    model_path.write_bytes(b"rl-model")
    monkeypatch.setenv("AI_TRADING_RL_RUNTIME_REQUIRE_GOVERNANCE", "1")
    monkeypatch.setenv("AI_TRADING_RL_RUNTIME_REQUIRE_PRODUCTION", "1")
    monkeypatch.setenv("AI_TRADING_RL_RUNTIME_REQUIRE_RECOMMEND", "1")
    monkeypatch.setenv("AI_TRADING_RL_RUNTIME_BOOTSTRAP_GOVERNANCE_SIDECAR", "1")

    loaded = bot_engine._reload_rl_agent_from_runtime_path(
        model_path=str(model_path),
        reason="runtime",
        use_rl_enabled=True,
        force=True,
    )

    assert loaded is False
    sidecar = Path(f"{model_path}.governance.json")
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["governance_status"] == "shadow"
    assert payload["recommend_use_rl_agent"] is False
