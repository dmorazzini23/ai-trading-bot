from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import ai_trading.execution.live_trading as lt


def test_submit_rate_limit_permit_times_out_after_burst(
    monkeypatch,
    tmp_path,
) -> None:
    engine: Any = lt.ExecutionEngine.__new__(lt.ExecutionEngine)
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
    engine_a: Any = lt.ExecutionEngine.__new__(lt.ExecutionEngine)
    engine_b: Any = lt.ExecutionEngine.__new__(lt.ExecutionEngine)

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
    engine: Any = lt.ExecutionEngine.__new__(lt.ExecutionEngine)
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
            setattr(
                exc,
                "http_error",
                SimpleNamespace(
                    response=SimpleNamespace(
                        status_code=429,
                        headers={"Retry-After": "2"},
                    )
                ),
            )
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


def test_submit_rate_limit_state_reader_clamps_and_ignores_bad_values(tmp_path) -> None:
    state_path = tmp_path / "submit_rate_state.json"
    state_path.write_text(
        json.dumps(
            {
                "tokens": "100",
                "last_refill_epoch": "-5",
                "cooldown_until_epoch": "nan",
            }
        ),
        encoding="utf-8",
    )

    state = lt.ExecutionEngine._read_submit_rate_limit_state(
        state_path,
        capacity=3,
        now_epoch=10.0,
    )

    assert state == {
        "tokens": 3.0,
        "last_refill_epoch": 0.0,
        "cooldown_until_epoch": 0.0,
    }


def test_submit_rate_limit_state_reader_uses_defaults_for_invalid_json(tmp_path) -> None:
    state_path = tmp_path / "submit_rate_state.json"
    state_path.write_text("{not-json", encoding="utf-8")

    state = lt.ExecutionEngine._read_submit_rate_limit_state(
        state_path,
        capacity=4,
        now_epoch=12.5,
    )

    assert state["tokens"] == 4.0
    assert state["last_refill_epoch"] == 12.5
    assert state["cooldown_until_epoch"] == 0.0


def test_submit_rate_limit_config_resolves_relative_state_path(monkeypatch, tmp_path) -> None:
    engine: Any = lt.ExecutionEngine.__new__(lt.ExecutionEngine)

    monkeypatch.setenv("AI_TRADING_ORDER_SUBMIT_RATE_LIMIT_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_ORDER_SUBMIT_RATE_LIMIT_PER_MIN", "120")
    monkeypatch.setenv("AI_TRADING_ORDER_SUBMIT_RATE_LIMIT_BURST", "5")
    monkeypatch.setenv("AI_TRADING_ORDER_SUBMIT_RATE_LIMIT_STATE_PATH", "state/submit.json")
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(tmp_path))

    config = engine._submit_rate_limit_config()

    assert config["enabled"] is True
    assert config["rpm"] == 120
    assert config["burst"] == 5
    assert config["state_path"] == tmp_path / "state" / "submit.json"
    assert str(config["lock_name"]).startswith("submit-rate-")


def test_reserve_submit_rate_limit_wait_reports_busy_lock(monkeypatch, tmp_path) -> None:
    engine: Any = lt.ExecutionEngine.__new__(lt.ExecutionEngine)

    class BusyLock:
        def __enter__(self):
            raise TimeoutError("busy")

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(lt, "process_file_lock", lambda *_a, **_k: BusyLock())

    wait_s, reason = engine._reserve_submit_rate_limit_wait(
        config={
            "state_path": tmp_path / "submit_rate_state.json",
            "lock_name": "busy-lock",
            "burst": 1,
            "refill_rate": 1.0,
        }
    )

    assert wait_s == 0.05
    assert reason == "submit_rate_lock_busy"


def test_acquire_submit_rate_limit_logs_one_wait_before_success(monkeypatch, caplog) -> None:
    engine: Any = lt.ExecutionEngine.__new__(lt.ExecutionEngine)
    waits = iter([(0.25, "submit_rate_limit"), (0.0, None)])
    sleeps: list[float] = []
    now_values = iter([100.0, 100.0, 100.1])

    engine._submit_rate_limit_config = lambda: {"enabled": True, "queue_timeout_sec": 1.0}
    engine._reserve_submit_rate_limit_wait = lambda *, config: next(waits)
    monkeypatch.setattr(lt, "monotonic_time", lambda: next(now_values))
    monkeypatch.setattr(lt.time, "sleep", lambda seconds: sleeps.append(float(seconds)))

    with caplog.at_level("INFO", logger="ai_trading.execution.live_trading"):
        ok, context = engine._acquire_submit_rate_limit_permit(
            symbol="MSFT",
            side="sell",
            order_type="market",
        )

    assert ok
    assert context == {"reason": None, "wait_s": 0.0}
    assert sleeps == [0.25]
    wait_records = [record for record in caplog.records if record.message == "ORDER_SUBMIT_RATE_LIMIT_WAIT"]
    assert len(wait_records) == 1
    assert getattr(wait_records[0], "symbol", None) == "MSFT"
