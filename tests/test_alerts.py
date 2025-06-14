import types
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import alerts


def test_alert_throttling(monkeypatch):
    sent = []
    monkeypatch.setattr(alerts, "SLACK_WEBHOOK", "http://example.com")
    monkeypatch.setattr(alerts.requests, "post", lambda *a, **k: sent.append(k))
    alerts.send_slack_alert("msg", key="err")
    alerts.send_slack_alert("msg", key="err")
    assert len(sent) == 1
    time.sleep(alerts.THROTTLE_SEC)
    alerts.send_slack_alert("msg", key="err")
    assert len(sent) == 2
