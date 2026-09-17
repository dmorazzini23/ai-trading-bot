"""Read-only health reporting for the scheduled market preflight.

Exit zero means a valid report was produced, not that trading is qualified.
Transport, HTTP protocol and malformed-payload failures exit one with JSON.
"""
from __future__ import annotations

import json
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

HEALTH_URL = "http://127.0.0.1:9001/healthz"


def collect_health() -> dict[str, Any]:
    """Read the canonical endpoint, retaining the body of readiness HTTP 503."""
    try:
        response = urlopen(HEALTH_URL, timeout=15)
    except HTTPError as exc:
        response = exc
    with response:
        status = response.code
        body = response.read(1_048_577)
    if len(body) > 1_048_576:
        raise ValueError("Health response exceeds one MiB")
    if status not in (200, 503):
        raise ValueError(f"Unexpected health HTTP status: {status}")
    payload = json.loads(body)
    if not isinstance(payload, dict) or not isinstance(payload.get("ok"), bool):
        raise ValueError("Health response must be an object with boolean ok")
    keys = (
        "status", "reason", "attention_flags", "readiness_failures",
        "readiness_gates", "replay_live_parity_gate", "service_state",
        "broker", "data_provider", "alpaca", "entry_control",
        "day_sleeve_model", "latest_training_attempt",
    )
    return {
        "report_status": "complete",
        "readiness_verdict": "ready" if status == 200 and payload["ok"] else "blocked",
        "http_status": status,
        "ok": payload["ok"],
        **{key: payload.get(key) for key in keys},
    }


def main() -> int:
    try:
        report = collect_health()
    except (URLError, OSError, ValueError) as exc:
        report = {"report_status": "failed", "readiness_verdict": "unknown",
                  "error_type": type(exc).__name__, "error": str(exc)}
        code = 1
    else:
        code = 0
    sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
