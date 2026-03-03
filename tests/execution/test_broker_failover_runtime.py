from __future__ import annotations

import json
from pathlib import Path

from ai_trading.execution import live_trading as lt


class _Adapter:
    provider = "paper"

    @staticmethod
    def submit_order(order_data):
        return {
            "id": "paper-1",
            "status": "accepted",
            "client_order_id": order_data.get("client_order_id"),
        }


def _engine_stub() -> lt.ExecutionEngine:
    engine = lt.ExecutionEngine.__new__(lt.ExecutionEngine)
    engine.stats = {}
    return engine


def test_attempt_failover_submit_success(monkeypatch, tmp_path: Path) -> None:
    engine = _engine_stub()
    playbook_path = tmp_path / "broker_playbook.jsonl"
    monkeypatch.setenv("AI_TRADING_BROKER_FAILOVER_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_BROKER_FAILOVER_PROVIDER", "paper")
    monkeypatch.setenv("AI_TRADING_BROKER_RESILIENCE_PLAYBOOK_PATH", str(playbook_path))
    monkeypatch.setattr(lt, "build_broker_adapter", lambda **_kwargs: _Adapter())

    response = engine._attempt_failover_submit(
        {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 2,
            "type": "limit",
            "limit_price": 190.0,
            "client_order_id": "cid-1",
        },
        primary_error=TimeoutError("primary broker timeout"),
    )

    assert response is not None
    assert response["failover"] is True
    assert response["provider"] == "paper"
    assert engine.stats["failover_submits"] == 1
    assert playbook_path.exists()
    lines = playbook_path.read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(lines[-1])
    assert payload["action"] == "failover_submit_success"


def test_attempt_failover_submit_disabled(monkeypatch) -> None:
    engine = _engine_stub()
    monkeypatch.setenv("AI_TRADING_BROKER_FAILOVER_ENABLED", "0")
    response = engine._attempt_failover_submit(
        {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 1,
            "type": "limit",
            "limit_price": 190.0,
            "client_order_id": "cid-2",
        },
        primary_error=TimeoutError("primary broker timeout"),
    )
    assert response is None
