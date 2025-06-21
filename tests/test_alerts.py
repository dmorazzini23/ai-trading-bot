import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import alerts


def test_alert_throttling(monkeypatch):
    """Verify repeated alerts are rate limited."""
    sent = []
    monkeypatch.setattr(alerts, "SLACK_WEBHOOK", "http://example.com")
    monkeypatch.setattr(alerts.requests, "post", lambda *a, **k: sent.append(k))
    monkeypatch.setattr(alerts, "_last_sent", {}, raising=False)
    monkeypatch.setattr(alerts, "THROTTLE_SEC", 0.1, raising=False)

    alerts.send_slack_alert("msg", key="err")
    alerts.send_slack_alert("msg", key="err")
    assert len(sent) == 1

    time.sleep(alerts.THROTTLE_SEC)

    alerts.send_slack_alert("msg", key="err")
    assert len(sent) == 2
