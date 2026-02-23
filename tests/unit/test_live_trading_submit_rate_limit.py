from __future__ import annotations

from types import SimpleNamespace

import ai_trading.execution.live_trading as lt


def test_submit_rate_limit_permit_times_out_after_burst(
    monkeypatch,
    tmp_path,
) -> None:
    engine = lt.ExecutionEngine.__new__(lt.ExecutionEngine)
    state_path = tmp_path / "submit_rate_state.json"

    monkeypatch.setenv("AI_TRADING_ORDER_SUBMIT_RATE_LIMIT_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_ORDER_SUBMIT_RATE_LIMIT_PER_MIN", "60")
    monkeypatch.setenv("AI_TRADING_ORDER_SUBMIT_RATE_LIMIT_BURST", "1")
    monkeypatch.setenv("AI_TRADING_ORDER_SUBMIT_RATE_LIMIT_QUEUE_TIMEOUT_SEC", "0")
    monkeypatch.setenv("AI_TRADING_ORDER_SUBMIT_RATE_LIMIT_STATE_PATH", str(state_path))

    first_ok, _ = engine._acquire_submit_rate_limit_permit(
        symbol="AAPL",
        side="buy",
        order_type="limit",
    )
    second_ok, second_ctx = engine._acquire_submit_rate_limit_permit(
        symbol="AAPL",
        side="buy",
        order_type="limit",
    )

    assert first_ok
    assert not second_ok
    assert second_ctx["reason"] == "submit_rate_limit"


def test_submit_rate_limit_retry_after_cooldown_is_shared(
    monkeypatch,
    tmp_path,
) -> None:
    state_path = tmp_path / "submit_rate_state.json"
    engine_a = lt.ExecutionEngine.__new__(lt.ExecutionEngine)
    engine_b = lt.ExecutionEngine.__new__(lt.ExecutionEngine)

    monkeypatch.setenv("AI_TRADING_ORDER_SUBMIT_RATE_LIMIT_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_ORDER_SUBMIT_RATE_LIMIT_PER_MIN", "120")
    monkeypatch.setenv("AI_TRADING_ORDER_SUBMIT_RATE_LIMIT_BURST", "10")
    monkeypatch.setenv("AI_TRADING_ORDER_SUBMIT_RATE_LIMIT_QUEUE_TIMEOUT_SEC", "0")
    monkeypatch.setenv("AI_TRADING_ORDER_SUBMIT_RATE_LIMIT_STATE_PATH", str(state_path))

    engine_a._apply_submit_rate_limit_cooldown(
        retry_after_seconds=30.0,
        symbol="AAPL",
        side="buy",
    )

    permit_ok, permit_ctx = engine_b._acquire_submit_rate_limit_permit(
        symbol="AAPL",
        side="buy",
        order_type="limit",
    )
    assert not permit_ok
    assert permit_ctx["reason"] == "broker_retry_after"


def test_execute_with_retry_honors_retry_after_header(monkeypatch) -> None:
    engine = lt.ExecutionEngine.__new__(lt.ExecutionEngine)
    engine.stats = {"retry_count": 0}
    engine.circuit_breaker = {"failure_count": 0, "is_open": False, "last_failure": None}
    engine._handle_nonretryable_api_error = lambda exc, *a, **k: None
    engine._handle_execution_failure = lambda exc: None
    engine._acquire_submit_rate_limit_permit = lambda **_kwargs: (True, {"reason": None, "wait_s": 0.0})

    cooldowns: list[float] = []
    engine._apply_submit_rate_limit_cooldown = (
        lambda **kwargs: cooldowns.append(float(kwargs["retry_after_seconds"]))
    )

    sleeps: list[float] = []
    monkeypatch.setattr(lt.time, "sleep", lambda seconds: sleeps.append(float(seconds)))

    attempts = {"count": 0}

    class _RateLimitedError(lt.APIError):
        pass

    def _submit_order_to_alpaca(order_data):
        attempts["count"] += 1
        if attempts["count"] == 1:
            exc = _RateLimitedError("rate limited")
            setattr(exc, "response", SimpleNamespace(status_code=429, headers={"Retry-After": "2"}))
            setattr(exc, "_status_code", 429)
            raise exc
        return {"status": "ok", "symbol": order_data["symbol"]}

    result = engine._execute_with_retry(
        _submit_order_to_alpaca,
        {"symbol": "AAPL", "side": "buy", "type": "limit"},
    )

    assert result == {"status": "ok", "symbol": "AAPL"}
    assert attempts["count"] == 2
    assert sleeps == [2.0]
    assert cooldowns == [2.0]
    assert engine.stats["retry_count"] == 1
