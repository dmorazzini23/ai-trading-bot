from __future__ import annotations
from datetime import datetime, timedelta, UTC
from typing import Optional
# AI-AGENT-REF: simple freshness checker for tests


def check_data_freshness(latest_ts: Optional[datetime], max_age: timedelta) -> bool:
    if latest_ts is None:
        return False
    return (datetime.now(UTC) - latest_ts) <= max_age
