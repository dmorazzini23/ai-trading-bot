from __future__ import annotations

import types

import requests
from ai_trading.net.http import TimeoutSession, build_retrying_session


def test_timeoutsession_injects_default_timeout(monkeypatch):
    captured = {}

    def fake_request(self, method, url, **kwargs):  # AI-AGENT-REF: capture timeout
        captured.update(kwargs)
        return types.SimpleNamespace(ok=True)

    monkeypatch.setattr(requests.Session, "request", fake_request, raising=True)

    s = TimeoutSession(default_timeout=(5.0, 10.0))
    s.request("GET", "http://unit.test")
    assert captured["timeout"] == (5.0, 10.0)

    captured.clear()
    s.request("GET", "http://unit.test", timeout=1.23)
    assert captured["timeout"] == 1.23


def test_build_retrying_session_defaults(monkeypatch):
    captured = {}

    def fake_request(self, method, url, **kwargs):  # AI-AGENT-REF: capture timeout
        captured.update(kwargs)
        return types.SimpleNamespace(ok=True)

    monkeypatch.setattr(requests.Session, "request", fake_request, raising=True)

    s = build_retrying_session(connect_timeout=2.0, read_timeout=3.0)
    s.request("GET", "http://unit.test")
    assert captured["timeout"] == (2.0, 3.0)

