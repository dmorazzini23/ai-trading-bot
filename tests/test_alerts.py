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

    alerts.send_slack_alert("msg", key="err", throttle_sec=0.1)
    alerts.send_slack_alert("msg", key="err", throttle_sec=0.1)
    assert len(sent) == 1

    time.sleep(0.1)

    alerts.send_slack_alert("msg", key="err", throttle_sec=0.1)
    assert len(sent) == 2
