"""Publish a small off-host liveness signal for the paper trading host."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from ai_trading.logging import get_logger


logger = get_logger(__name__)
NAMESPACE = "AITrading/Host"
METRIC_NAME = "PaperRuntimeResponding"
HOST_ID = "ai-trading-primary"
HEALTH_URL = "http://127.0.0.1:9001/healthz"


def _health_responding(*, timeout: float = 15.0) -> bool:
    """A structured degraded reply proves liveness, but never trading readiness."""

    try:
        with urlopen(HEALTH_URL, timeout=timeout) as response:
            code = response.status
            payload = json.load(response)
    except HTTPError as exc:
        code = exc.code
        try:
            payload = json.load(exc)
        except (ValueError, OSError):
            return False
    except (URLError, TimeoutError, ValueError, OSError):
        return False
    return code in {200, 503} and isinstance(payload, dict) and payload.get("status") in {
        "healthy", "ready", "degraded",
    }


def collect_liveness() -> int:
    """Return 1 only when systemd and the canonical local health route answer."""

    try:
        active = subprocess.run(
            ["systemctl", "is-active", "--quiet", "ai-trading.service"],
            check=False,
            timeout=5,
            capture_output=True,
        ).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        active = False
    return int(active and _health_responding())


def publish_liveness(value: int, *, region: str = "us-east-2") -> None:
    """Send one standard-resolution metric; an omitted heartbeat also alarms."""

    if value not in (0, 1):
        raise ValueError("liveness must be 0 or 1")
    datum: dict[str, Any] = {
        "MetricName": METRIC_NAME,
        "Dimensions": [{"Name": "Host", "Value": HOST_ID}],
        "Timestamp": datetime.now(UTC).isoformat(),
        "Unit": "Count",
        "Value": value,
    }
    subprocess.run(
        [
            "aws", "cloudwatch", "put-metric-data", "--region", region,
            "--namespace", NAMESPACE, "--metric-data", json.dumps([datum]),
        ],
        check=True,
        timeout=20,
        capture_output=True,
    )


def main() -> int:
    value = collect_liveness()
    try:
        publish_liveness(value)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.error(
            "HOST_HEARTBEAT_PUBLISH_FAILED",
            extra={"reason": type(exc).__name__, "responding": bool(value)},
        )
        return 1
    logger.info("HOST_HEARTBEAT_PUBLISHED", extra={"responding": bool(value)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
