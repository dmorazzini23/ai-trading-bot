import types
import sys
from pathlib import Path

import pytest

import alerting


def test_send_slack_no_webhook(monkeypatch, caplog):
    """Warning is logged when webhook is missing."""
    monkeypatch.setattr(alerting.settings, "SLACK_WEBHOOK", "", raising=False)
    caplog.set_level("WARNING")
    alerting.send_slack_alert("msg")
    assert "SLACK_WEBHOOK not set" in caplog.text


def test_send_slack_success(monkeypatch):
    """Slack POST request is issued when webhook exists."""
    called = {}
    def fake_post(url, json, timeout):
        called['url'] = url
        called['json'] = json
        called['timeout'] = timeout
    monkeypatch.setattr(alerting.settings, "SLACK_WEBHOOK", "http://hook", raising=False)
    monkeypatch.setattr(alerting.requests, "post", fake_post)
    alerting.send_slack_alert("hello")
    assert called['url'] == "http://hook" and called['json']['text'] == "hello"
