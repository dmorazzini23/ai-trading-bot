from __future__ import annotations

import signal as _signal
import types

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

