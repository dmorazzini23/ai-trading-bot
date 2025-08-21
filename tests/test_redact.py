from __future__ import annotations

# Tests for redaction utility.  # AI-AGENT-REF
from ai_trading.logging.redact import redact


def test_redact_masks_nested() -> None:
    payload = {
        "apiKey": "abc",
        "nested": {"secret": "def", "token": "ghi"},
        "pwd": "not-masked",  # key not matched
    }
    out = redact(payload)
    assert out["apiKey"] == "***REDACTED***"
    assert out["nested"]["secret"] == "***REDACTED***"
    assert out["nested"]["token"] == "***REDACTED***"
    assert out["pwd"] == "not-masked"

