from __future__ import annotations

import errno
import threading
import types

import pytest

import ai_trading.main as main


def test_check_alpaca_sdk_strict_missing_exits(monkeypatch) -> None:
    monkeypatch.setattr(main, "ALPACA_AVAILABLE", False)
    monkeypatch.setattr(main, "should_enforce_strict_import_preflight", lambda: True)

    with pytest.raises(SystemExit) as exc:
        main._check_alpaca_sdk()

    assert exc.value.code == 1


def test_assert_singleton_api_detects_existing_health(monkeypatch) -> None:
    monkeypatch.setattr(main, "_managed_env", lambda name, default=None: "")
    monkeypatch.setattr(main, "sys", types.SimpleNamespace(modules={}))
    monkeypatch.setattr(main, "get_pid_on_port", lambda port: None)
    monkeypatch.setattr(main, "_probe_local_api_health", lambda port, **kwargs: True)

    with pytest.raises(main.ExistingApiDetected):
        main._assert_singleton_api(types.SimpleNamespace(env="prod", api_port=9001))


def test_start_api_with_signal_records_exception(monkeypatch) -> None:
    ready = threading.Event()
    error = threading.Event()
    exc = OSError(errno.EADDRINUSE, "busy")

    monkeypatch.setattr(main, "should_stop", lambda: False)
    monkeypatch.setattr(main, "start_api", lambda _ready: (_ for _ in ()).throw(exc))

    main.start_api_with_signal(ready, error)

    assert not ready.is_set()
    assert error.is_set()
    assert getattr(error, "exception") is exc


def test_wait_for_api_startup_catches_late_thread_error(monkeypatch) -> None:
    class _Ready:
        def wait(self, timeout: float | None = None) -> bool:
            _ = timeout
            return False

        def is_set(self) -> bool:
            return False

    class _LateError:
        def __init__(self, exc: BaseException) -> None:
            self.exception = exc
            self._calls = 0

        def wait(self, timeout: float | None = None) -> bool:
            _ = timeout
            self._calls += 1
            return self._calls >= 2

    class _AliveThread:
        def is_alive(self) -> bool:
            return True

    exc = OSError(errno.EADDRINUSE, "busy")
    monkeypatch.setenv("AI_TRADING_API_STARTUP_WAIT_SECONDS", "6")

    with pytest.raises(OSError) as raised:
        main._wait_for_api_startup(
            _Ready(),
            _LateError(exc),
            _AliveThread(),
            initial_wait_seconds=5.0,
        )

    assert raised.value is exc


def test_interruptible_sleep_non_test_stops_after_first_slice(monkeypatch) -> None:
    slept: list[float] = []
    stop_states = iter([False, True])
    monkeypatch.setattr(main, "_is_test_mode", lambda: False)
    monkeypatch.setattr(main, "should_stop", lambda: next(stop_states))
    monkeypatch.setattr(main.time, "sleep", lambda seconds: slept.append(seconds))

    main._interruptible_sleep(2.0)

    assert slept == [0.25]


def test_validate_runtime_config_collects_multiple_failures(monkeypatch) -> None:
    monkeypatch.setattr(main, "_get_equity_from_alpaca", lambda _cfg: None)
    monkeypatch.setattr(main, "resolve_max_position_size", lambda *args, **kwargs: (100.0, {}))

    cfg = types.SimpleNamespace(
        trading_mode="wild",
        alpaca_base_url="https://api.alpaca.markets",
        paper=True,
    )
    settings = types.SimpleNamespace(capital_cap=1.5, dollar_risk_limit=0.0)

    with pytest.raises(ValueError) as exc:
        main._validate_runtime_config(cfg, settings)

    message = str(exc.value)
    assert "AI_TRADING_TRADING_MODE invalid: wild" in message
    assert "AI_TRADING_CAPITAL_CAP out of range: 1.5" in message
    assert "DOLLAR_RISK_LIMIT out of range: 0.0" in message
    assert "paper endpoint" in message
