import types
import sys
import pytest
import os

# Ensure repository root in path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("APCA_API_KEY_ID", "dummy")
os.environ.setdefault("APCA_API_SECRET_KEY", "dummy")

# stub missing deps
sys.modules.setdefault('requests', types.SimpleNamespace(post=lambda *a, **k: None))
for _m in ['dotenv']:
    mod = types.ModuleType(_m)
    if _m == 'dotenv':
        mod.load_dotenv = lambda *a, **k: None
    sys.modules.setdefault(_m, mod)

import alerts
import alpaca_api
import slippage
import rebalancer


def test_send_slack_alert(monkeypatch):
    messages = []
    monkeypatch.setattr(alerts, "SLACK_WEBHOOK", "http://example.com")
    def fake_post(url, json, timeout=5):
        messages.append((url, json))
    monkeypatch.setattr(alerts.requests, "post", fake_post)
    alerts.send_slack_alert("hi")
    assert messages and messages[0][1]["text"] == "hi"


def test_submit_order_shadow(monkeypatch):
    class DummyAPI:
        def submit_order(self, order_data=None):
            raise AssertionError("should not call in shadow")
    monkeypatch.setattr(alpaca_api, "SHADOW_MODE", True)
    log = types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    resp = alpaca_api.submit_order(DummyAPI(), types.SimpleNamespace(symbol="AAPL", qty=1, side="buy", time_in_force="day"), log)
    assert resp["status"] == "shadow"


def test_monitor_slippage_alert(monkeypatch):
    alerts_sent = []
    monkeypatch.setattr(slippage, "SLIPPAGE_THRESHOLD", 0.001)
    monkeypatch.setattr(slippage, "send_slack_alert", lambda m: alerts_sent.append(m))
    slippage.monitor_slippage(100.0, 102.0, "AAPL")
    assert alerts_sent


def test_maybe_rebalance(monkeypatch):
    calls = []
    monkeypatch.setattr(rebalancer, "REBALANCE_INTERVAL_MIN", 0)
    monkeypatch.setattr(rebalancer, "rebalance_portfolio", lambda ctx: calls.append(ctx))
    rebalancer._last_rebalance = rebalancer.datetime.now(rebalancer.timezone.utc) - rebalancer.timedelta(minutes=1)
    rebalancer.maybe_rebalance("ctx")
    assert calls == ["ctx"]
