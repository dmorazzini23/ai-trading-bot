from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace


def _run_import_check(source: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTEST_RUNNING"] = "1"
    return subprocess.run(
        [sys.executable, "-c", source],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )


def test_sentiment_import_does_not_resolve_session_or_settings():
    proc = _run_import_check(
        """
import ai_trading.config.settings as settings
import ai_trading.net.http as http

def boom_settings():
    raise RuntimeError("settings resolved at import")

def boom_session():
    raise RuntimeError("session resolved at import")

settings.get_settings = boom_settings
http.get_http_session = boom_session

from ai_trading.analysis import sentiment

assert sentiment._http_session is None
assert sentiment.SENTIMENT_MAX_RETRIES == 5
assert sentiment.SENTIMENT_BACKOFF_BASE == 5.0
"""
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout


def test_predict_import_does_not_resolve_http_session():
    proc = _run_import_check(
        """
import ai_trading.net.http as http

def boom_session():
    raise RuntimeError("session resolved at import")

http.get_http_session = boom_session

import ai_trading.predict as predict

assert not hasattr(predict, "_HTTP")
"""
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout


def test_sentiment_retry_policy_resolves_fresh_settings(monkeypatch):
    from ai_trading.analysis import sentiment

    settings = [
        SimpleNamespace(
            sentiment_max_retries=2,
            sentiment_backoff_base=1.5,
            sentiment_backoff_strategy="fixed",
        ),
        SimpleNamespace(
            sentiment_max_retries=7,
            sentiment_backoff_base=3.0,
            sentiment_backoff_strategy="exponential",
        ),
    ]
    monkeypatch.setattr(sentiment, "get_settings", lambda: settings.pop(0))

    assert sentiment._current_sentiment_retry_policy() == (2, 1.5, "fixed")
    assert sentiment._current_sentiment_retry_policy() == (7, 3.0, "exponential")
