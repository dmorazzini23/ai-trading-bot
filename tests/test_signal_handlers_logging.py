from __future__ import annotations

import signal as _signal
import types
from typing import Any

from ai_trading import main


def test_install_signal_handlers_logs_service_signal(monkeypatch):
    calls = []

    def fake_info(msg, *args, **kwargs):  # AI-AGENT-REF: capture SERVICE_SIGNAL
        calls.append((msg, args, kwargs))

    monkeypatch.setattr(main, "logger", types.SimpleNamespace(info=fake_info))

    installed = {}

    def fake_signal(sig, handler):  # AI-AGENT-REF: record handler
        installed[sig] = handler

    monkeypatch.setattr(_signal, "signal", fake_signal)

    main._install_signal_handlers()
    assert _signal.SIGINT in installed and _signal.SIGTERM in installed

    handler = installed[_signal.SIGINT]
    handler(_signal.SIGINT, None)

    assert any(call[0] == "SERVICE_SIGNAL" for call in calls)


def test_shutdown_runtime_resources_uses_nonblocking_cleanup(monkeypatch):
    calls = []

    from ai_trading.core import executors
    from ai_trading.utils import workers

    monkeypatch.setattr(
        executors,
        "cleanup_executors",
        lambda *, wait: calls.append(("core", wait)),
    )
    monkeypatch.setattr(
        workers,
        "shutdown_all",
        lambda *, wait: calls.append(("workers", wait)),
    )
    monkeypatch.setattr(
        main._logging,
        "shutdown_queue_listener",
        lambda *, timeout: calls.append(("logging", timeout)),
        raising=False,
    )

    main._shutdown_runtime_resources(wait=False)

    assert ("core", False) in calls
    assert ("workers", False) in calls
    assert ("logging", 1.0) in calls


def test_shutdown_force_exit_timer_is_armed_once(monkeypatch):
    calls: list[tuple[Any, ...]] = []

    class FakeTimer:
        daemon = False

        def __init__(self, seconds, callback, args=()):
            calls.append(("timer", seconds, callback, args))

        def start(self):
            calls.append(("start", self.daemon))

    monkeypatch.setattr(main, "_SHUTDOWN_FORCE_EXIT_TIMER", None)
    monkeypatch.setattr(main, "_is_test_mode", lambda: False)
    monkeypatch.setattr(main, "_shutdown_force_exit_seconds", lambda: 7.0)
    monkeypatch.setattr(main.threading, "Timer", FakeTimer)
    monkeypatch.setattr(
        main,
        "logger",
        types.SimpleNamespace(info=lambda *args, **kwargs: calls.append(("log", args, kwargs))),
    )

    main._arm_shutdown_force_exit("SIGTERM")
    main._arm_shutdown_force_exit("SIGTERM")

    assert calls.count(("start", True)) == 1
    assert sum(1 for call in calls if call[0] == "timer") == 1
